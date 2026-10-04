#=
    Environments that store one of every pair of hermitian-conjugate bond states.

    For a hermitian hamiltonian, and bra = ket, the environment of a bond state ā whose operators are the hermitian
    conjugates of those of a is the conjugate of a's: bra and ket swapped, the MPO leg flipped, up to a scalar
    (_conjenv). So per bond only one of every such pair is stored; the other is rebuilt when an environment is asked
    for, and the transfers only compute the stored states (the MPO tensor restricted to them on its output side).
    The effective operators are MPSKit's own, on the rebuilt environments.

    Which bond states pair up, and with which scalar, is known when the hamiltonian is built (for quantum chemistry
    from the builder's labels, see quantum_chemistry_hamiltonian(...; paired = true)); a PairedHamiltonian carries
    it. check_conjugate_pairs verifies a pairing against the environments of a random state.
=#

const BlockTensorKit = MPSKit.BlockTensorKit     # not a dependency of this package, MPSKit loads it

# MPSKit's own environments of ψ for H: newer versions take (ψ, H, ψ), older ones keep a third argument as a fixed
# `above` state, so there it has to be (ψ, H)
_mpskit_environments(ψ,H) = applicable(environments,ψ,H) ? environments(ψ,H) : environments(ψ,H,ψ)

# the environment of the hermitian-conjugate bond state: bra and ket swapped (the adjoint with the MPO leg as a
# spectator), and the MPO leg flipped to the conjugate sector
function _conjenv(t)
    r = TensorKit.permute(t,((1,),(3,2)))'
    M = space(r,2)
    f = isomorphism(storagetype(t),flip(M),M)
    @planar out[-1 -2; -3] := f[-2; 1]*r[-1 1; -3]
    out
end

# per bond state (in MPSKit's order): the stored state it is rebuilt from (itself if it is stored) and the scalar,
# env[a] = scale[a] * conjenv(env[from[a]])
struct ConjugatePairs
    from::Vector{Int}
    scale::Vector{Float64}
    stored::Vector{Int}
    position::Vector{Int}       # per bond state: its position among the stored ones (0 if not stored)
end
function ConjugatePairs(from,scale)
    stored = findall(from .== eachindex(from))
    position = zeros(Int,length(from)); position[stored] .= eachindex(stored)
    ConjugatePairs(from,scale,stored,position)
end
ConjugatePairs(n::Int) = ConjugatePairs(collect(1:n),ones(n))

"""
    PairedHamiltonian(H, lpairs, rpairs)

A hermitian `FiniteMPOHamiltonian` with, per bond, which bond states are hermitian conjugates of each other in the
left (`lpairs`) and right (`rpairs`) environments. `find_groundstate`, `environments` and `expectation_value` use
`paired_environments` for it.
"""
struct PairedHamiltonian{O}
    H::O
    lpairs::Vector{ConjugatePairs}
    rpairs::Vector{ConjugatePairs}
end
Base.length(P::PairedHamiltonian) = length(P.H)
Base.getindex(P::PairedHamiltonian,i) = P.H[i]
MPSKit.physicalspace(P::PairedHamiltonian,i::Int) = physicalspace(P.H,i)
MPSKit.left_virtualspace(P::PairedHamiltonian,i::Int) = left_virtualspace(P.H,i)
MPSKit.right_virtualspace(P::PairedHamiltonian,i::Int) = right_virtualspace(P.H,i)

MPSKit.environments(ψ::FiniteMPS,P::PairedHamiltonian,args...;kwargs...) = paired_environments(ψ,P)
MPSKit.expectation_value(ψ::FiniteMPS,P::PairedHamiltonian,envs...) = expectation_value(ψ,P.H,envs...)
MPSKit.AC_hamiltonian(pos::Int,below,P::PairedHamiltonian,above,envs;kwargs...) = MPSKit.AC_hamiltonian(pos,below,P.H,above,envs;kwargs...)
MPSKit.AC2_hamiltonian(pos::Int,below,P::PairedHamiltonian,above,envs;kwargs...) = MPSKit.AC2_hamiltonian(pos,below,P.H,above,envs;kwargs...)

mutable struct PairedEnvironments{O,C,L,R} <: MPSKit.AbstractMPSEnvironments
    operator::O
    lops::Vector{Any}                   # per site: the MPO tensor with its right states restricted to the stored ones
    rops::Vector{Any}                   # per site: the MPO tensor with its left states restricted to the stored ones
    lpairs::Vector{ConjugatePairs}      # per bond
    rpairs::Vector{ConjugatePairs}
    ldependencies::Vector{C}
    rdependencies::Vector{C}
    GLs::Vector{L}                      # per bond, only the stored states
    GRs::Vector{R}
end

"""
    paired_environments(ψ, P::PairedHamiltonian) -> PairedEnvironments

Environments that store and compute only one of every pair of hermitian-conjugate bond states; the others are
rebuilt when asked for.
"""
function paired_environments(ψ::FiniteMPS,P::PairedHamiltonian)
    N = length(ψ); H = P.H
    sparseW(n) = BlockTensorKit.SparseBlockTensorMap(H[n])
    lops = Any[sparseW(n)[:,:,:,P.lpairs[n+1].stored] for n in 1:N]
    rops = Any[sparseW(n)[P.rpairs[n].stored,:,:,:] for n in 1:N]
    menv = _mpskit_environments(ψ,H)
    GL1 = menv.GLs[1]; GRN = menv.GRs[end]
    t = similar(ψ.AL[1])
    PairedEnvironments(H,lops,rops,P.lpairs,P.rpairs,fill(t,N),fill(t,N),
                       [b == 1 ? GL1 : similar(GL1) for b in 1:N+1],[b == N+1 ? GRN : similar(GRN) for b in 1:N+1])
end

# a random state with the spaces of ψ's physical legs and boundaries, and every allowed sector with multiplicity up
# to `mult` on the bonds: generic environment blocks, for check_conjugate_pairs
function _probe_state(ψ;mult = 4)
    N = length(ψ)
    P = [space(ψ.AL[i],2) for i in 1:N]
    cap(V) = typeof(V)(s => min(mult,dim(V,s)) for s in sectors(V))
    V = [left_virtualspace(ψ,1)]
    for i in 1:N-1
        push!(V,cap(fuse(V[end]⊗P[i])))
    end
    FiniteMPS(randn,scalartype(ψ.AL[1]),P,V[2:end];left = left_virtualspace(ψ,1),right = right_virtualspace(ψ,N))
end

"""
    check_conjugate_pairs(P::PairedHamiltonian, ψ) -> worst relative residual

Checks env[c] = scale * conjenv(env[from[c]]) for every pair, on the full environments of `ψ` (take a random state
with generic blocks). Pairs whose environments both vanish are skipped.
"""
function check_conjugate_pairs(P::PairedHamiltonian,ψ::FiniteMPS)
    N = length(ψ)
    menv = _mpskit_environments(ψ,P.H)
    worst = 0.0
    for b in 2:N, (side,pairs) in ((:left,P.lpairs[b]),(:right,P.rpairs[b]))
        G = side === :left ? leftenv(menv,b,ψ) : rightenv(menv,b-1,ψ)
        for c in eachindex(pairs.from)
            a = pairs.from[c]; a == c && continue
            ta = G[1,a,1]; tc = G[1,c,1]
            max(norm(ta),norm(tc)) < 1e-12 && continue
            worst = max(worst,norm(tc - pairs.scale[c]*_conjenv(ta))/max(norm(ta),norm(tc)))
        end
    end
    worst
end

# the full environment of a bond from its stored states (mpospace() gives the MPO space of the bond)
function _expand(stored,pairs::ConjugatePairs,mpospace,side)
    length(pairs.stored) == length(pairs.from) && return stored
    V = side === :left ? (codomain(stored)[1] ⊗ mpospace()' ← domain(stored)) : (codomain(stored)[1] ⊗ mpospace() ← domain(stored))
    full = similar(stored,V)
    for a in eachindex(pairs.from)
        p = pairs.position[pairs.from[a]]
        full[1,a,1] = pairs.from[a] == a ? stored[1,p,1] : pairs.scale[a]*_conjenv(stored[1,p,1])
    end
    full
end

function MPSKit.poison!(ca::PairedEnvironments,ind)
    ca.ldependencies[ind] = similar(ca.ldependencies[ind])
    ca.rdependencies[ind] = similar(ca.rdependencies[ind])
end

function MPSKit.leftenv(ca::PairedEnvironments,ind,state;kwargs...)
    a = findfirst(i -> !(state.AL[i] === ca.ldependencies[i]),1:(ind-1))
    if !isnothing(a)
        for j in a:(ind-1)
            GL = _expand(ca.GLs[j],ca.lpairs[j],() -> left_virtualspace(ca.operator,j),:left)
            ca.GLs[j+1] = GL*MPSKit.TransferMatrix(state.AL[j],ca.lops[j],state.AL[j])
            ca.ldependencies[j] = state.AL[j]
        end
    end
    return _expand(ca.GLs[ind],ca.lpairs[ind],() -> left_virtualspace(ca.operator,ind),:left)
end

function MPSKit.rightenv(ca::PairedEnvironments,ind,state;kwargs...)
    a = findfirst(i -> !(state.AR[i] === ca.rdependencies[i]),length(state):-1:(ind+1))
    if !isnothing(a)
        a = length(state)-a+1
        for j in a:-1:(ind+1)
            GR = _expand(ca.GRs[j+1],ca.rpairs[j+1],() -> right_virtualspace(ca.operator,j),:right)
            ca.GRs[j] = MPSKit.TransferMatrix(state.AR[j],ca.rops[j],state.AR[j])*GR
            ca.rdependencies[j] = state.AR[j]
        end
    end
    return _expand(ca.GRs[ind+1],ca.rpairs[ind+1],() -> right_virtualspace(ca.operator,ind),:right)
end
