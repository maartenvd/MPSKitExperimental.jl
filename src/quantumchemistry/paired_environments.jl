#=
    Paired environments store part of the environments implicitly. 
    It's often possible that an environment channel can be written as the conjugate of another (or a linear combination!), as long as the bra and ket states are the same.
    This environment implements that optimization.

    It works for a PairedHamiltonian, which is an MPOHamiltonian which carries this extra information.
=#

const BlockTensorKit = MPSKit.BlockTensorKit     # not a dependency of this package, MPSKit loads it

# the environment of the hermitian-conjugate bond state: bra and ket swapped (the adjoint with the MPO leg as a
# spectator), and the MPO leg flipped to the conjugate sector
function _conjenv(t)
    r = TensorKit.permute(t,((1,),(3,2)))'
    M = space(r,2)
    f = isomorphism(storagetype(t),flip(M),M)
    @planar out[-1 -2; -3] := f[-2; 1]*r[-1 1; -3]
    out
end

# we can often get parts of the environments without computing their transfer, as long as bra == ket, through conjugation
# To compute the full environment, we only need to hold a reduced one holding select computed state.
# Per state a of the full environment, either from[a] == a (a is computed), or env[a] = scale[a] * conjenv(env[from[a]]).
# stored and position are precomputed metadata: 
#     stored[k] maps slot k of the reduced environment to its state in the full one
#     position is the inverse (0 for states that are not computed).
struct ConjugatePairs
    from::Vector{Int}
    scale::Vector{Float64}

    # metadata
    stored::Vector{Int}
    position::Vector{Int}
end
function ConjugatePairs(from,scale)
    # constructor that precomputes metadata
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

#=
    The effective operators.

    MPSKit splits them by Jordan block. 
    The large block sums over the continuing states m of the middle bond (the bond between the two sites for AC2, the right bond for AC):
        y = Σ_m L_m x R_m, 
    with 
        L_m = GL·W1[:,:,:,m]
        R_m = W2[m,:,:,:]·GR 
    (L_m = GL·W[:,:,:,m] and R_m = GR[m] for AC), precomputed for every m.

    Here L_m and R_m are only precomputed for the middle states in the reduced environment.
        
    A rebuilt middle state c with partner a = from[c] contributes the adjoint of a's term, applied on the fly from a's L_a and R_a:
        L_c x R_c = w_c L_a' x R_a',    w_c = ls[c] rs[c] θ_c
    with ls, rs the scales of c in the left and right environments, and θ_c coming from the two conjenv flippers.
    
    By construction, our hamiltonian doesn't mix (L_a' x R_a) blocks, we only have 
        L_a x R_a + w_a L_a' x R_a'
    Those cross terms would fall back to slower code.
=#
function MPSKit.AC_hamiltonian(site::Int,below,P::PairedHamiltonian,above,envs;prepare::Bool = true)
    GL = leftenv(envs,site,below)
    GR = rightenv(envs,site,below)

    W = P.H[site]

    # this object is some lazy 'precomputed version'
    H = MPSKit.JordanMPO_AC_Hamiltonian(GL,W,GR)

    states = _paired_states(P,site+1)
    
    # if there are paired states and A is nontrivial, then we'll create an optimized operator.
    if !ismissing(H.A) && !isnothing(states)
        H = _with_continuing(H,PairedContinuing(GL[2:end-1],W.A,nothing,GR[2:end-1],states...))
    end
    
    return prepare ? MPSKit.prepare_operator!!(H) : H
end
function MPSKit.AC2_hamiltonian(site::Int,below,P::PairedHamiltonian,above,envs;prepare::Bool = true)
    GL = leftenv(envs,site,below)
    GR = rightenv(envs,site+1,below)


    W1 = P.H[site]
    W2 = P.H[site+1]

    # the full mpskit 'lazily contracted object'
    H = MPSKit.JordanMPO_AC2_Hamiltonian(GL,W1,W2,GR)

    # check if there are pairing terms that we can exploit
    states = _paired_states(P,site+1)
    if !ismissing(H.AA) && !isnothing(states)
        H = _with_continuing(H,PairedContinuing(GL[2:end-1],W1.A,W2.A,GR[2:end-1],states...))
    end
    
    return prepare ? MPSKit.prepare_operator!!(H) : H
end

_with_continuing(H::MPSKit.JordanMPO_AC_Hamiltonian{O1,O2},A) where {O1,O2} =
    MPSKit.JordanMPO_AC_Hamiltonian{O1,O2,typeof(A)}(H.D,H.I,H.E,H.C,H.B,A)
_with_continuing(H::MPSKit.JordanMPO_AC2_Hamiltonian{O1,O2,O3},AA) where {O1,O2,O3} =
    MPSKit.JordanMPO_AC2_Hamiltonian{O1,O2,O3,typeof(AA)}(H.II,H.IC,H.ID,H.CB,H.CA,H.AB,AA,H.BE,H.DE,H.EE)


function _paired_states(P::PairedHamiltonian,b)
    lp = P.lpairs[b]
    rp = P.rpairs[b]

    # "from" is either itself, or points to the virtual index that the environment can be calculated from
    # we only - for now - support hamiltonians with the 'lp.from == rp.from" property otherwise _paired_states returns nothing, and we fall back to the 'slow' plain mpskit fallback
    (lp.from == rp.from && length(lp.from) > 2) || return nothing
    
    # the size of the full environments
    n = length(lp.from)
    V = left_virtualspace(P.H,b)
    
    weight = zeros(length(lp.stored))
    for c in 1:n # loop through the unreduced basis
        # this means - c is calculated from c itself
        a = lp.from[c]; a == c && continue

        # only the 'A' part of the MPOHamiltonian is precontracted, and so this must hold (holds for qchem)
        (2 <= a <= n-1 && 2 <= c <= n-1) || return nothing

        # if weight is nonzero, it means that two environments that are impliclty defined (through conjenv) need to be contracted, with weight:
        weight[lp.position[a]] += lp.scale[c]*rp.scale[c]*twist(only(sectors(V[c])))
    end

    return (lp.stored[2:end-1] .- 1,weight[2:end-1])
end

# the continuing block before preparing: GL, W1.A, W2.A (nothing for one site), GR, the stored states and their
# weights
struct PairedContinuing{L,O1,O2,R}
    GL::L
    W1::O1
    W2::O2
    GR::R
    stored::Vector{Int}
    weight::Vector{Float64}
end

# per group of stored states with the same weight: the weight, and MPSKit's own prepared operator over those states
struct PreparedPaired{D}
    weights::Vector{Float64}
    ops::Vector{D}
end

# MPSKit's (unpaired) continuing block, restricted to some of the stored states
_continuing(H::PairedContinuing{L,O1,Nothing,R},states) where {L,O1,R} =
    MPSKit.MPO_AC_Hamiltonian(H.GL,H.W1[:,:,:,states],H.GR[states])
_continuing(H::PairedContinuing,states) = MPSKit.MPO_AC2_Hamiltonian(H.GL,H.W1[:,:,:,states],H.W2[states,:,:,:],H.GR)

MPSKit.prepared_operator_type(::Type{PairedContinuing{L,O1,Nothing,R}},::Type{B},::Type{A}) where {L,O1,R,B,A} =
    PreparedPaired{MPSKit.prepared_operator_type(MPSKit.MPO_AC_Hamiltonian{L,O1,R},B,A)}
MPSKit.prepared_operator_type(::Type{PairedContinuing{L,O1,O2,R}},::Type{B},::Type{A}) where {L,O1,O2,R,B,A} =
    PreparedPaired{MPSKit.prepared_operator_type(MPSKit.MPO_AC2_Hamiltonian{L,O1,O2,R},B,A)}

function MPSKit.prepare_operator!!(H::PairedContinuing,backend::MPSKit.AbstractBackend,allocator)
    P = MPSKit.prepared_operator_type(typeof(H),typeof(backend),typeof(allocator))

    weights = unique(H.weight)
    ops = fieldtype(P,:ops)()
    for w in weights
        # find all reduced parts that share weight w
        states = H.stored[H.weight .== w]
        push!(ops,MPSKit.prepare_operator!!(_continuing(H,states),backend,allocator))
    end
    P(weights,ops)
end

(H::PairedContinuing)(x) = MPSKit.prepare_operator!!(H)(x)     # unprepared use (MPSKit's toolbox)

#---------------------
function (H::PreparedPaired)(x)
    y = similar(x,TensorOperations.promote_contract(scalartype(x),scalartype(H.ops[1])))
    for g in eachindex(H.ops)
        _add_group!(y,H.ops[g],x,H.weights[g],g > 1)      # the first group overwrites y
    end
    y
end

#=
    y = β y + D x + w D' x, for MPSKit's prepared operator D of one group.

    In its fused layout (x a matrix per sector u), D x = L · (x ∘ R), with x ∘ R MPSKit's mul_front!: x acting on the
    first leg of R, subblock by subblock. The forward term is done as MPSKit does it, but added into y.
    The adjoint is Z = L' · x, followed by the transpose of mul_front!: per subblock (f₁,f₂) of Z, with u the first
    uncoupled sector and c the coupled one,
        y_u += dim(c)/dim(u) · Z[f₁,f₂] ⋅ conj(R[f₁,f₂])        (contracting all but the first leg)
    mul_front! needs no dimension factor; its transpose does, since the inner product weighs a block by its dimension.
    Z has the space of x ∘ R, so it reuses that intermediate's buffer; fuse_legs reuses the data, so everything
    accumulates straight into y.
=#
_add_group!(y,D::MPSKit.PrecomputedAC2Derivative,x,w,β) =
    _add_group_fused!(MPSKit.fuse_legs(y,2,2),MPSKit.fuse_legs(D.leftenv,2,1),MPSKit.fuse_legs(D.rightenv,1,2),
                      MPSKit.fuse_legs(x,2,2),w,β,D.backend,D.allocator)
_add_group!(y,D::MPSKit.PrecomputedACDerivative,x,w,β) =
    _add_group_fused!(MPSKit.fuse_legs(y,2,1),MPSKit.fuse_legs(D.leftenv,2,1),D.rightenv,
                      MPSKit.fuse_legs(x,2,1),w,β,D.backend,D.allocator)
function _add_group_fused!(y,L,R,x,w,β,backend,allocator)
    cp = allocator_checkpoint!(allocator)
    TC = TensorOperations.promote_contract(scalartype(x),scalartype(L),scalartype(R))
    xR = TensorOperations.tensoralloc_contract(TC,x,((1,),(2,)),false,R,((1,),(2,3)),false,((1,2),(3,)),
                                               Val(true),allocator)

    MPSKit.mul_front!(xR,x,R,true,false,backend,allocator)
    mul!(y,L,xR,true,β)

    if !iszero(w)
        Z = mul!(xR,L',x)
        Rstructure = TensorKit.subblockstructure(space(R))
        for ((f₁,f₂),z) in subblocks(Z)
            haskey(Rstructure,(f₁,f₂)) || continue
            u = first(f₁.uncoupled)
            TensorOperations.tensorcontract!(block(y,u),z,((1,),(2,3)),false,R[f₁,f₂],((2,3),(1,)),true,((1,),(2,)),
                                             w*dim(f₁.coupled)/dim(u),true,backend,allocator)
        end
    end

    TensorOperations.tensorfree!(xR,allocator)
    allocator_reset!(allocator,cp)
    y
end

struct PairedEnvironments{O,W,C,L,R} <: MPSKit.AbstractMPSEnvironments
    operator::O
    
    lops::Vector{W}                     # per site: the MPO tensor with its right states restricted to the stored ones
    rops::Vector{W}                     # per site: the MPO tensor with its left states restricted to the stored ones
    
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
    N = length(ψ); 
    H = P.H
    
    sparseW(n) = BlockTensorKit.SparseBlockTensorMap(H[n])
    
    # this would not have been as easy without the existence of blocktensorkit!!!
    lops = [sparseW(n)[:,:,:,P.lpairs[n+1].stored] for n in 1:N]
    rops = [sparseW(n)[P.rpairs[n].stored,:,:,:] for n in 1:N]
    
    menv = environments(ψ,H,ψ)
    
    GL1 = menv.GLs[1]; 
    GRN = menv.GRs[end]
    
    t = similar(ψ.AL[1])
    
    PairedEnvironments(H,lops,rops,P.lpairs,P.rpairs,fill(t,N),fill(t,N),
                       [b == 1 ? GL1 : similar(GL1) for b in 1:N+1],[b == N+1 ? GRN : similar(GRN) for b in 1:N+1])
end


# the full environment of a bond from its reduced one
function _expand(reduced,pairs::ConjugatePairs,mpospace,side)
    length(pairs.stored) == length(pairs.from) && return reduced
    V = codomain(reduced)[1] ⊗ (side === :left ? mpospace' : mpospace) ← domain(reduced)
    full = similar(reduced,V)
    for a in eachindex(pairs.from)
        k = pairs.position[pairs.from[a]]
        full[1,a,1] = pairs.from[a] == a ? reduced[1,k,1] : pairs.scale[a]*_conjenv(reduced[1,k,1])
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
            GL = _expand(ca.GLs[j],ca.lpairs[j],left_virtualspace(ca.operator,j),:left)
            ca.GLs[j+1] = GL*MPSKit.TransferMatrix(state.AL[j],ca.lops[j],state.AL[j])
            ca.ldependencies[j] = state.AL[j]
        end
    end
    return _expand(ca.GLs[ind],ca.lpairs[ind],left_virtualspace(ca.operator,ind),:left)
end

function MPSKit.rightenv(ca::PairedEnvironments,ind,state;kwargs...)
    a = findfirst(i -> !(state.AR[i] === ca.rdependencies[i]),length(state):-1:(ind+1))
    if !isnothing(a)
        a = length(state)-a+1
        for j in a:-1:(ind+1)
            GR = _expand(ca.GRs[j+1],ca.rpairs[j+1],right_virtualspace(ca.operator,j),:right)
            ca.GRs[j] = MPSKit.TransferMatrix(state.AR[j],ca.rops[j],state.AR[j])*GR
            ca.rdependencies[j] = state.AR[j]
        end
    end
    return _expand(ca.GRs[ind+1],ca.rpairs[ind+1],right_virtualspace(ca.operator,ind),:right)
end
