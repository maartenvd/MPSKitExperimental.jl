# Rather than re-deriving the fusion-tree recoupling data ourselves (which, for
# non-`UniqueFusion` sectors, requires the same pack/matmul/unpack machinery TensorKit
# now implements internally), we precompute TensorKit's own cached `TreeTransformer`
# and reuse its (already optimal) application kernel at call time.

# for `Trivial` sectortype (no symmetry, dense tensors) TensorKit skips fusion trees
# entirely and works directly on the reshaped dense array
struct DensePermute end

function generate_permute_table(elt,sp_src,sp_dst, p1::IndexTuple{N₁},p2::IndexTuple{N₂}) where {N₁,N₂}
    p = (p1,p2)
    plin = (TensorKit.linearize(p), ()) # only the linear permutation matters for the array kernels
    sectortype(sp_src) === Trivial && return (DensePermute(),plin)
    levels = (TensorKit.codomainind(sp_src)..., TensorKit.domainind(sp_src)...)
    transformer = TensorKit.treebraider(sp_dst, sp_src, p, false, levels)
    (transformer,plin)
end

execute_permute_table!(t_dst,t_src,bulk,alpha=true,beta=false,allocator=TensorOperations.DefaultAllocator()) =
    _execute_transform_table!(t_dst,t_src,bulk,alpha,beta,allocator)

function _execute_transform_table!(t_dst,t_src,bulk::Tuple{DensePermute,Any},alpha,beta,allocator)
    (_,plin) = bulk
    TensorOperations.tensoradd!(t_dst[],t_src[],plin,false,alpha,beta,
        TensorOperations.DefaultBackend(),allocator)
    t_dst
end

function _execute_transform_table!(t_dst,t_src,bulk,alpha,beta,allocator)
    (transformer,plin) = bulk
    dst = TensorKit.StridedSubblocks(t_dst, transformer.structure_dst)
    src = TensorKit.StridedSubblocks(t_src, transformer.structure_src)
    TensorKit.add_transform_kernel!(dst,src,plin,false,transformer,alpha,beta,
        TensorOperations.DefaultBackend(),allocator,1)
    t_dst
end




# tensoradd

function create_mediated_tensoradd!(C, pC, A, conjA, α=1, β=1 , backend=nothing)
    (C,Nothing)
end

function mediated_tensoradd!(fst,mediator,args...)
    TensorOperations.tensoradd!(args...)
end

# tensoralloc_add
function create_mediated_tensoralloc_add(TC, A::SymbolicTensorMap, pC::Index2Tuple{N₁,N₂}, conjA, istemp=Val(false), backend = TensorOperations.DefaultAllocator())  where {N₁,N₂}

    S = spacetype(ttype(A))

    spaces1 = [conjA ? conj(A.structure[p]) : A.structure[p] for p in pC[1]]
    spaces2 = [conjA ? conj(A.structure[p]) : A.structure[p] for p in pC[2]]
    cod = ProductSpace{S,N₁}(spaces1...)
    dom = ProductSpace{S,N₂}(conj.(spaces2)...)
    stortype = TensorKit.similarstoragetype(ttype(A),TC)
    C = SymbolicTensorMap(tensormaptype(S,N₁, N₂, stortype),dom → cod)

    (C,fast_init(cod,dom,stortype))
end

function mediated_tensoralloc_add(fst,mediator,TC, A, pC::Index2Tuple{N₁,N₂},conjA, istemp=Val(false), backend= TensorOperations.DefaultAllocator())  where {N₁,N₂}
    mediator(fst.allocator,istemp)
end

# tensortrace 
function create_mediated_tensortrace!(C, pC, A, pA, conjA, α=1, β=0 , backend=nothing)
    (C,Nothing)
end

function mediated_tensortrace!(fst,mediator,args...)
    TensorOperations.tensortrace!(args...)
end

#Base.conj(P::ProductSpace) = ProductSpace(map(conj, P.spaces))

# tensorcontract
function create_mediated_tensorcontract!(C::SymbolicTensorMap, A::SymbolicTensorMap, pA, conjA, B::SymbolicTensorMap, pB, conjB, pC,  α=1, β=0 , b1=nothing, b2=nothing)
    S = spacetype(A.structure)
    bstyle = BraidingStyle(sectortype(S))
    if !(bstyle isa SymmetricBraiding)
        throw(SectorMismatch("only tensors with symmetric braiding rules can be contracted; try `@planar` instead"))
    end
    # dense tensors: like TensorKit, contract the underlying arrays directly with a single
    # TensorOperations call instead of permute -> mul! -> permute on TensorMaps
    sectortype(S) === Trivial && return (C, TrivialContract())
    # the permutation kernels below only conjugate the spaces, not the data
    (conjA || conjB) && throw(ArgumentError("tightloop does not support conj(...) in contractions of tensors with symmetries"))

    A_structure = !conjA ? A.structure : conj(codomain(A.structure)) ← conj(domain(A.structure))
    B_structure = !conjB ? B.structure : conj(codomain(B.structure)) ← conj(domain(B.structure))

    # like TensorKit.contract!: sort the contracted indices along A or B, and contract A*B or
    # B*A, whichever needs the fewest permutations. Only the identity permutation is free for
    # tensors with fusion trees, so this only depends on the spaces and is decided once here.
    pA′, pB′, pA″, pB″, pC′ = TensorKit._contract_candidates(pA, pB, pC)
    candidates = ((false, pA′, pB′, pC), (false, pA″, pB″, pC),
                  (true, reverse(pB′), reverse(pA′), pC′), (true, reverse(pB″), reverse(pA″), pC′))
    costs = map(candidates) do (swap, p1, p2, p3)
        swap ? _contract_cost(C.structure, B_structure, p1, A_structure, p2, p3, bstyle) :
               _contract_cost(C.structure, A_structure, p1, B_structure, p2, p3, bstyle)
    end
    (swap, p1, p2, p3) = candidates[argmin(costs)]
    plan = swap ? _contract_plan(C, B, B_structure, p1, A, A_structure, p2, p3, bstyle, true) :
                  _contract_plan(C, A, A_structure, p1, B, B_structure, p2, p3, bstyle, false)
    (C, plan)
end

_isidentityperm(sp, p) = p[1] == ntuple(identity, length(codomain(sp))) &&
    p[2] == ntuple(n -> n + length(codomain(sp)), length(domain(sp)))

# Fermionic sectors need an extra sign correction (a `twist`) of either A or B whenever any of
# B's contracted legs is dual. Mirrors TensorKit's `blas_contract!`: twist the operand that is
# copied anyway, or the smallest one if neither or both are. A twist can never be applied
# in-place on an input tensor, so it forces a copy even when the permutation is a no-op.
function _permutes_and_twists(A_structure, pA, B_structure, pB, bstyle)
    permA = !_isidentityperm(A_structure, pA)
    permB = !_isidentityperm(B_structure, pB)
    if bstyle isa Fermionic && any(n -> isdual(B_structure[n]), pB[1])
        twistA = !(permA ⊻ permB) ? dim(A_structure) < dim(B_structure) : permA
        twistB = !twistA
    else
        twistA = twistB = false
    end
    return (permA | twistA, permB | twistB, twistA, twistB)
end

function _contract_cost(C_structure, A_structure, pA, B_structure, pB, pC, bstyle)
    permA, permB, _, _ = _permutes_and_twists(A_structure, pA, B_structure, pB, bstyle)
    permC = !(pC[1] == ntuple(identity, length(pA[1])) && pC[2] == ntuple(n -> n + length(pA[1]), length(pB[2])))
    return dim(A_structure) * permA + dim(B_structure) * permB + dim(C_structure) * permC
end

struct ContractPlan{FA,TA,FB,TB,FC,TC}
    swap::Bool # contract B*A instead of A*B
    permA::Bool; fast_init_A::FA; tbl_A::TA
    permB::Bool; fast_init_B::FB; tbl_B::TB
    permC::Bool; fast_init_C′::FC; tbl_C′::TC
    twistA::Bool; twistB::Bool
end

function _contract_plan(C, A, A_structure, pA, B, B_structure, pB, pC, bstyle, swap)
    S = spacetype(A_structure)

    #A′ = permute(A, (oindA, cindA))
    sp_dst_A =  ProductSpace{S,length(pA[1])}(map(n -> A_structure[n], pA[1])) ← ProductSpace{S,length(pA[2])}(map(n -> conj(A_structure[n]), pA[2]))
    fast_init_A = fast_init(codomain(sp_dst_A),domain(sp_dst_A),storagetype(ttype(A)))
    tbl_A = generate_permute_table(scalartype(ttype(A)),A_structure,sp_dst_A,pA[1],pA[2])

    #B′ = permute(B, (cindB, oindB))
    sp_dst_B =  ProductSpace{S,length(pB[1])}(map(n -> B_structure[n], pB[1])) ← ProductSpace{S,length(pB[2])}(map(n -> conj(B_structure[n]), pB[2]))
    fast_init_B = fast_init(codomain(sp_dst_B),domain(sp_dst_B),storagetype(ttype(B)))
    tbl_B = generate_permute_table(scalartype(ttype(B)),B_structure,sp_dst_B,pB[1],pB[2])

    fast_init_C′ = fast_init(codomain(sp_dst_A),domain(sp_dst_B),storagetype(ttype(C)));
    tbl_C′ = generate_permute_table(scalartype(ttype(C)),codomain(sp_dst_A)←domain(sp_dst_B),C.structure,pC[1],pC[2])

    permA, permB, twistA, twistB = _permutes_and_twists(A_structure, pA, B_structure, pB, bstyle)
    permC = !(pC[1] == ntuple(identity, length(pA[1])) && pC[2] == ntuple(n -> n + length(pA[1]), length(pB[2])))

    ContractPlan(swap, permA, fast_init_A, tbl_A, permB, fast_init_B, tbl_B, permC, fast_init_C′, tbl_C′, twistA, twistB)
end

struct TrivialContract end

function mediated_tensorcontract!(fst,::TrivialContract,C, A, pA, conjA, B, pB, conjB, pC, α=1, β=0 , b1=nothing, b2=nothing)
    TensorOperations.tensorcontract!(C[], A[], pA, conjA, B[], pB, conjB, pC, α, β,
        TensorOperations.DefaultBackend(), fst.allocator)
    C
end

function mediated_tensorcontract!(fst,plan::ContractPlan,C, A, pA, conjA, B, pB, conjB, pC, α=1, β=0 , b1=nothing, b2=nothing)
    plan.swap ? _execute_contract_plan!(fst, plan, C, B, A, α, β) :
                _execute_contract_plan!(fst, plan, C, A, B, α, β)
end

function _execute_contract_plan!(fst, plan, C, A, B, α, β)
    if plan.permA
        Ap = plan.fast_init_A(fst.allocator,Val(true))
        execute_permute_table!(Ap,A,plan.tbl_A,true,false,fst.allocator)
        plan.twistA && twist!(Ap, filter(n -> !isdual(space(Ap,n)), domainind(Ap)))
    else
        Ap = A
    end

    if plan.permB
        Bp = plan.fast_init_B(fst.allocator,Val(true))
        execute_permute_table!(Bp,B,plan.tbl_B,true,false,fst.allocator)
        plan.twistB && twist!(Bp, filter(n -> isdual(space(Bp,n)), codomainind(Bp)))
    else
        Bp = B
    end

    if plan.permC
        C′ = mul!(plan.fast_init_C′(fst.allocator,Val(true)),Ap,Bp,α)
        execute_permute_table!(C,C′,plan.tbl_C′,true,β,fst.allocator)
        tensorfree!(C′, fst.allocator)
    else
        mul!(C,Ap,Bp,α,β)
    end

    plan.permA && tensorfree!(Ap, fst.allocator)
    plan.permB && tensorfree!(Bp, fst.allocator)

    C
end


# tensoralloc_contract
function create_mediated_tensoralloc_contract(TC, A::SymbolicTensorMap, pA, conjA, B::SymbolicTensorMap, pB, conjB, pC::Index2Tuple{N₁,N₂}, istemp, backend=TensorOperations.DefaultAllocator())  where {N₁,N₂}
    spaces1 = [conjA ? conj(A.structure[p]) : A.structure[p] for p in pA[1]]
    spaces2 = [conjB ? conj(B.structure[p]) : B.structure[p] for p in pB[2]]
    spaces = (spaces1..., spaces2...)

    S = spacetype(ttype(A))
    cod = ProductSpace{S,N₁}(getindex.(Ref(spaces), pC[1]))
    dom = ProductSpace{S,N₂}(conj.(getindex.(Ref(spaces), pC[2])))
    stortype = TensorKit.similarstoragetype(ttype(A),TC)
    C = SymbolicTensorMap(tensormaptype(S,N₁, N₂, stortype),dom → cod)

    (C,fast_init(cod,dom,stortype)) 
end

function mediated_tensoralloc_contract(fst,mediator,TC, A, pA, conjA, B, pB, conjB, pC::Index2Tuple{N₁,N₂}, istemp, backend=TensorOperations.DefaultAllocator())  where {N₁,N₂}
    mediator(fst.allocator,istemp)
end