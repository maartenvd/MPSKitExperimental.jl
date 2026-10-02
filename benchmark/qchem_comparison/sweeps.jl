include("common.jl")

(; st, NORB, hams) = load_data()
hams["fused_norr"] = hams["fused"]
mpos = envsetting("MPOS", "fused_jordan fused fused_norr opsum")
repeats = parse(Int, get(ENV, "REPEATS", "1"))
warmup = get(ENV, "WARMUP", "1") == "1"
timers = get(ENV, "TIMERS", "0") == "1"
header()

# one sweep per run, always starting from the same state; verbosity 4 makes DMRG2 print its timer
alg = DMRG2(; trunc = truncrank(BOND), maxiter = 1, tol = 1.0e-14, verbosity = timers ? 4 : 1)

function onesweep(k)
    H = hams[k]
    MPSKitExperimental.AC2_RANKREDUCE[] = k != "fused_norr"
    try
        t = @elapsed (ψ, envs) = find_groundstate(copy(st), H, alg)
        return t, real(expectation_value(ψ, H, envs))
    finally
        MPSKitExperimental.AC2_RANKREDUCE[] = true
    end
end

results = map(mpos) do k
    # the first run in a process compiles; timing it overstated the fused/MPSKit gap in earlier comparisons
    warmup && onesweep(k)
    runs = [onesweep(k) for _ in 1:repeats]
    t = minimum(first, runs)
    @printf("%-13s %8.1f s per sweep   E = %.10f   load = %.1f\n", k, t, last(first(runs)), first(Sys.loadavg()))
    k => t
end

ref = Dict(results)[first(mpos)]
println("\nrelative to ", first(mpos), ":")
for (k, t) in results
    @printf("%-13s %6.2f×\n", k, t / ref)
end
