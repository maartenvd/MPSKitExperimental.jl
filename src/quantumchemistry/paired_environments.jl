#=
    Environments that store one of every pair of hermitian-conjugate bond states.

    For a hermitian hamiltonian, and bra = ket, the environment of a bond state ā whose operators are the hermitian
    conjugates of those of a is the conjugate of a's: bra and ket swapped, the MPO leg flipped, up to a scalar
    (_conjenv). So per bond only one of every such pair is stored; the other is rebuilt when an environment is asked
    for, and the transfers only compute the stored states (the MPO tensor restricted to them on its output side).
    The effective operators are MPSKit's own, on the rebuilt environments.

    Which states pair up, and with which scalar, is found once from the environments of two random probe states
    with generic (not too small) blocks: a pair has to match, uniquely and with the same scalar, in both. States
    without such a partner (self-conjugate ones, or ones whose conjugate is a combination of several states) are
    stored. Bond states that carry several sectors usually only pair per sector, so the hamiltonian should have a
    single sector per bond state (quantum_chemistry_hamiltonian(...; split_sectors = true)).
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

# per bond state: the stored state it is rebuilt from (itself if it is stored) and the scalar: env[a] = scale[a]*conj(env[from[a]])
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

# probes[p][a]: the environment of bond state a in probe state p
function _conjugate_pairs(probes)
    n = length(first(probes))
    conj_probes = [[norm(t) > 1e-12 ? _conjenv(t) : nothing for t in blocks] for blocks in probes]
    partner = zeros(Int,n); λs = zeros(n)
    for a in 1:n
        cands = Tuple{Int,Float64}[]
        for c in 1:n
            l = NaN; ok = true
            for (blocks,cblocks) in zip(probes,conj_probes)
                r = cblocks[a]; u = blocks[c]
                (isnothing(r) || norm(u) <= 1e-12 || space(u) != space(r)) && (ok = false; break)
                λ = real(dot(r,u)/dot(r,r))
                norm(u - λ*r) <= 1e-10*norm(u) || (ok = false; break)
                isnan(l) ? (l = λ) : (isapprox(l,λ;rtol = 1e-8) || (ok = false; break))
            end
            ok && push!(cands,(c,l))
        end
        length(cands) == 1 && ((partner[a],λs[a]) = only(cands))
    end
    from = collect(1:n); scale = ones(n)
    for a in 1:n
        c = partner[a]
        (c > a && partner[c] == a) || continue
        from[c] = a; scale[c] = λs[a]
    end
    ConjugatePairs(from,scale)
end

# a random state with the spaces of ψ's physical legs and boundaries, and every allowed sector with multiplicity
# up to `mult` on the bonds: generic environment blocks for finding the pairs
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
    paired_environments(ψ, H) -> PairedEnvironments

Environments for `find_groundstate` (and `expectation_value`) that store and compute only one of every pair of
hermitian-conjugate bond states of `H` (a hermitian `FiniteMPOHamiltonian`, ideally with a single sector per bond
state, see `quantum_chemistry_hamiltonian(...; split_sectors = true)`).
"""
function paired_environments(ψ::FiniteMPS,H::FiniteMPOHamiltonian;mult::Int = 4)
    N = length(ψ)
    probes = [_probe_state(ψ;mult) for _ in 1:2]
    penvs = [_mpskit_environments(p,H) for p in probes]
    lpairs = [b == 1 ? ConjugatePairs(1) : _conjugate_pairs([[GL[1,a,1] for a in 1:size(GL,2)] for GL in (leftenv(e,b,p) for (e,p) in zip(penvs,probes))])
              for b in 1:N]
    push!(lpairs,ConjugatePairs(1))
    rpairs = [b == N+1 || b == 1 ? ConjugatePairs(1) :
              _conjugate_pairs([[GR[1,a,1] for a in 1:size(GR,2)] for GR in (rightenv(e,b-1,p) for (e,p) in zip(penvs,probes))])
              for b in 1:N+1]
    sparseW(n) = BlockTensorKit.SparseBlockTensorMap(H[n])
    lops = Any[sparseW(n)[:,:,:,lpairs[n+1].stored] for n in 1:N]
    rops = Any[sparseW(n)[rpairs[n].stored,:,:,:] for n in 1:N]
    menv = _mpskit_environments(ψ,H)
    GL1 = menv.GLs[1]; GRN = menv.GRs[end]
    t = similar(ψ.AL[1])
    PairedEnvironments(H,lops,rops,lpairs,rpairs,fill(t,N),fill(t,N),
                       [b == 1 ? GL1 : similar(GL1) for b in 1:N+1],[b == N+1 ? GRN : similar(GRN) for b in 1:N+1])
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
