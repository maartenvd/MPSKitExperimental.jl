#=
    A hermitian hamiltonian through half of it: H = h + h†.

    In environments where bra and ket are the same state, the effective operator of h† is the adjoint of the
    effective operator of h, so DMRG only needs h: its environments (fewer bond states, so less to transfer and
    store) and its precomputed effective operators, applied as h_eff + h_eff†. The adjoint reuses MPSKit's
    precomputed form: y = Σ_m L_m x R_m (with the middle leg m on L's domain and R's codomain) has the adjoint
    y ↦ Σ_m L_m† y R_m†, i.e. the same form with L̃[a; o, m] = conj(L[o; a, m]) and R̃[b, m; c] = conj(R[c, m; b]),
    which are transposes of L' and R' computed once per operator.

    For quantum chemistry h is the half of the paths of H that leave the start state into a positively charged
    bond state, plus half of those that leave it into a neutral one (see quantum_chemistry_hamiltonian).
=#

"""
    HermitianHalf(h)

The hermitian hamiltonian `h + h'`, represented by `h` (a `FiniteMPOHamiltonian`). It can be used with
`find_groundstate` (DMRG/DMRG2), `environments`, `disk_environments` and `expectation_value`.
"""
struct HermitianHalf{O}
    half::O
end
Base.length(H::HermitianHalf) = length(H.half)
MPSKit.physicalspace(H::HermitianHalf,i::Int) = physicalspace(H.half,i)

MPSKit.environments(below::FiniteMPS,H::HermitianHalf,args...;kwargs...) = environments(below,H.half,args...;kwargs...)
disk_environments(state::FiniteMPS,H::HermitianHalf) = disk_environments(state,H.half)
MPSKit.expectation_value(ψ::FiniteMPS,H::HermitianHalf,envs...) = 2real(expectation_value(ψ,H.half,envs...))

# h_eff + h_eff†
struct HermitianPair{P,Q}
    P::P
    Q::Q
end
(H::HermitianPair)(x) = add!(H.P(x),H.Q(x))
Base.:*(H::HermitianPair,x) = H(x)

function MPSKit.AC2_hamiltonian(pos::Int,below,H::HermitianHalf,above,envs;
                                backend = TensorOperations.DefaultBackend(),allocator = TensorOperations.DefaultAllocator(),kwargs...)
    h = H.half
    P = _precomputed(leftenv(envs,pos,below),(h[pos],h[pos+1]),rightenv(envs,pos+1,below),backend,allocator)
    return HermitianPair(P,AdjointPrecomputed(P))
end
function MPSKit.AC_hamiltonian(pos::Int,below,H::HermitianHalf,above,envs;
                               backend = TensorOperations.DefaultBackend(),allocator = TensorOperations.DefaultAllocator(),kwargs...)
    P = _precomputed(leftenv(envs,pos,below),(H.half[pos],),rightenv(envs,pos,below),backend,allocator)
    return HermitianPair(P,AdjointPrecomputed(P))
end

# MPSKit's generic precomputed form (GL·O and O·GR), in either MPSKit version this package runs with: newer ones
# keep backend and allocator in the operator, older ones take them when preparing
function _precomputed(GL,Os,GR,backend,allocator)
    mk = length(Os) == 2 ? MPSKit.MPO_AC2_Hamiltonian : MPSKit.MPO_AC_Hamiltonian
    if hasmethod(mk,Tuple{typeof.((GL,Os...,GR,backend,allocator))...})
        MPSKit.prepare_operator!!(mk(GL,Os...,GR,backend,allocator))
    else
        MPSKit.prepare_operator!!(mk(GL,Os...,GR),backend,allocator)
    end
end

# the adjoint of a precomputed (single or two site) effective operator, applied with MPSKit's own code on the
# fused (bond tensor) level
struct AdjointPrecomputed{LT,RT,B,A}
    L::LT
    R::RT
    backend::B
    allocator::A
end
function AdjointPrecomputed(P::MPSKit.PrecomputedDerivative)
    Lf = MPSKit.fuse_legs(P.leftenv,2,1)
    Rf = numin(P.rightenv) == 2 ? MPSKit.fuse_legs(P.rightenv,1,2) : P.rightenv
    AdjointPrecomputed(transpose(Lf',((1,),(3,2))),transpose(Rf',((1,3),(2,))),P.backend,P.allocator)
end
function (Q::AdjointPrecomputed)(y)
    yf = MPSKit.fuse_legs(y,2,numin(y))
    xf = MPSKit.PrecomputedDerivative(Q.L,Q.R,Q.backend,Q.allocator)(yf)
    TensorMap{scalartype(xf)}(xf.data,space(y))
end
