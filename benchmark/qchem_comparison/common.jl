using LinearAlgebra, Printf, Serialization
using TensorKit, MPSKit, MPSKitExperimental
using BlockTensorKit: nonzero_pairs, nonzero_length

const EXAMPLES = normpath(joinpath(@__DIR__, "..", "..", "examples"))
include(joinpath(EXAMPLES, "opsum_qchem_su2.jl"))
using .OpSumQChemSU2

# all comparisons are single-threaded BLAS unless asked otherwise; the fused code threads over julia threads only
LinearAlgebra.BLAS.set_num_threads(parse(Int, get(ENV, "BLAS_THREADS", "1")))

const FCIDUMP = get(ENV, "FCIDUMP", "N2.STO3G.FCIDUMP")
const BOND = parse(Int, get(ENV, "BOND", "50"))
const DATAFILE = joinpath(@__DIR__, "data", "$(splitext(FCIDUMP)[1])_D$(BOND).jls")

envsetting(name, default) = split(get(ENV, name, default))

header() = @printf("# %s, D = %d, julia threads = %d, BLAS threads = %d, load = %.1f\n",
    FCIDUMP, BOND, Threads.nthreads(), LinearAlgebra.BLAS.get_num_threads(), first(Sys.loadavg()))

"""
    realify(H::FiniteMPOHamiltonian)

OpSum always emits `ComplexF64` tensors; comparing those against a real MPO mixes in the cost of complex arithmetic.
"""
function realify(H::FiniteMPOHamiltonian)
    Ws = map(parent(H)) do W
        O = MPSKit.jordanmpotensortype(spacetype(W), Vector{Float64})(undef, space(W))
        # `nonzero_pairs(W)` would materialise the identity scalars as tensors
        for (ix, v) in nonzero_pairs(W.tensors)
            norm(imag(v)) <= 1.0e-12 * max(norm(v), 1) || error("OpSum MPO has a complex entry at $ix")
            O[ix] = real(v)
        end
        for (ix, c) in W.scalars
            abs(imag(c)) <= 1.0e-12 * max(abs(c), 1) || error("OpSum MPO has a complex identity at $ix")
            O[ix] = real(c)
        end
        O
    end
    return FiniteMPOHamiltonian(Ws)
end

"""
    load_data() -> (; st, NORB, hams)

The reference state written by `setup.jl` and every MPO variant, keyed by name:

- `opsum`: the OpSum `BipartiteAlgorithm` MPO, real, as `JordanMPOTensor`s
- `opsum_complex`: the same as OpSum emits it (`ComplexF64`), as in the original notebook
- `fused`: Maarten's `LinkMPOHamiltonian` (operators on the sites, scalar links)
- `fused_jordan`: the same hamiltonian converted to `JordanMPOTensor`s on the same bond states, i.e. MPSKit's format
"""
function load_data()
    isfile(DATAFILE) || error("no data for $FCIDUMP at D = $BOND: run setup.jl first")
    data = deserialize(DATAFILE)
    hams = Dict{String, Any}(
        "opsum_complex" => data.ham_opsum,
        "opsum" => realify(data.ham_opsum),
        "fused" => data.ham_fused,
        "fused_jordan" => FiniteMPOHamiltonian(data.ham_fused),
    )
    return (; data.st, data.NORB, hams)
end

isfused(H) = H isa MPSKitExperimental.LinkMPOHamiltonian
fused_environments(st, H) = isfused(H) ? disk_environments(st, H) : environments(st, H, st)

function mpo_summary(io::IO, hams, NORB)
    names = sort!(collect(keys(hams)))
    bonddims = Dict(k => (isfused(H) ? [length(H.bondspaces[i + 1]) for i in 1:NORB] : [length(right_virtualspace(H, i).spaces) for i in 1:NORB],
                          isfused(H) ? [sum(dim, H.bondspaces[i + 1]) for i in 1:NORB] : [dim(right_virtualspace(H, i)) for i in 1:NORB])
                    for (k, H) in hams)
    entries(H, i) = isfused(H) ? length(H.channels[i]) : nonzero_length(H[i].tensors)
    println(io, "per bond: #bond states / total dimension; per site: genuine operator entries (channels for `fused`)")
    @printf(io, "%5s", "site")
    for k in names
        @printf(io, " %26s", k)
    end
    println(io)
    for i in 1:NORB
        @printf(io, "%5d", i)
        for k in names
            nch, d = bonddims[k]
            @printf(io, " %26s", "$(nch[i]) / $(d[i]) | $(entries(hams[k], i))")
        end
        println(io)
    end
    return nothing
end
