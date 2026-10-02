include("common.jl")
using BenchmarkTools
using TensorOperations: BufferAllocator
using MPSKitExperimental: fused_AC2_hamiltonian

(; st, NORB, hams) = load_data()
mpos = envsetting("MPOS", "opsum fused_jordan fused")
positions = parse.(Int, envsetting("POS", string(NORB ÷ 2)))
rounds = parse(Int, get(ENV, "ROUNDS", "3"))
budget = parse(Float64, get(ENV, "BUDGET", "2"))
header()

envs = Dict(k => fused_environments(st, hams[k]) for k in mpos)

# the cases timed for one MPO at one bond, as name => closure; DMRG2 uses a BufferAllocator for its sweep
function cases(k, pos, x)
    H, E = hams[k], envs[k]
    if isfused(H)
        GL = MPSKit.leftenv(E, pos, st)
        Hrr = fused_AC2_hamiltonian(pos, st, H, E; rankreduce = true)
        Hnorr = fused_AC2_hamiltonian(pos, st, H, E; rankreduce = false)
        return [
            "$k build" => () -> fused_AC2_hamiltonian(pos, st, H, E; rankreduce = false),
            "$k build (rank red.)" => () -> fused_AC2_hamiltonian(pos, st, H, E; rankreduce = true),
            "$k apply" => () -> Hnorr * x,
            "$k apply (rank red.)" => () -> Hrr * x,
            "$k transfer_left" => () -> MPSKit.transfer_left(GL, H[pos], st.AL[pos], st.AL[pos]),
        ]
    else
        allocator = BufferAllocator()
        GL = MPSKit.leftenv(E, pos, st)
        Heff = MPSKit.AC2_hamiltonian(pos, st, H, st, E; allocator)
        return [
            "$k build" => () -> MPSKit.AC2_hamiltonian(pos, st, H, st, E; allocator),
            "$k apply" => () -> Heff * x,
            "$k transfer_left" => () -> GL * MPSKit.TransferMatrix(st.AL[pos], H[pos], st.AL[pos]; allocator),
        ]
    end
end

for pos in positions
    for k in mpos
        MPSKit.leftenv(envs[k], pos, st)
        MPSKit.rightenv(envs[k], pos + 1, st)
    end
    x = MPSKit.AC2(st, pos; kind = :ACAR)
    all_cases = reduce(vcat, [cases(k, pos, x) for k in mpos])

    @printf("## bond %d, D = (%d, %d)\n", pos, dim(left_virtualspace(st, pos)), dim(right_virtualspace(st, pos + 1)))
    applies = [(name, f()) for (name, f) in all_cases if occursin("apply", name)]
    y_ref = last(first(applies))
    for (name, y) in applies[2:end]
        @printf("|y - y(%s)| / |y| for %s: %.1e\n", first(first(applies)), name, norm(y - y_ref) / norm(y_ref))
    end

    allocs = Dict(name => (f(); @allocated f()) for (name, f) in all_cases)
    best = Dict(name => Inf for (name, _) in all_cases)
    medians = Dict(name => Inf for (name, _) in all_cases)
    # interleaved rounds, keeping the minimum: robust against other jobs on a shared machine
    for _ in 1:rounds, (name, f) in all_cases
        b = run(@benchmarkable $f() samples = 50 evals = 1 seconds = budget)
        best[name] = min(best[name], minimum(b).time / 1.0e6)
        medians[name] = min(medians[name], median(b).time / 1.0e6)
    end
    @printf("%-32s %12s %12s %12s\n", "case", "min [ms]", "median [ms]", "alloc [MiB]")
    for (name, _) in all_cases
        @printf("%-32s %12.2f %12.2f %12.1f\n", name, best[name], medians[name], allocs[name] / 2^20)
    end
    @printf("# load after bond %d: %.1f\n", pos, first(Sys.loadavg()))
end
