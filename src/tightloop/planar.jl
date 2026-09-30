# transpose
function generate_transpose_table(elt,sp_src,sp_dst, p1::IndexTuple{N₁},p2::IndexTuple{N₂}) where {N₁,N₂}
    p = (p1,p2)
    plin = (TensorKit.linearize(p), ())
    sectortype(sp_src) === Trivial && return (DensePermute(),plin)
    transformer = TensorKit.treetransposer(sp_dst, sp_src, p, false)
    (transformer,plin)
end

execute_transpose_table!(t_dst,t_src,bulk,alpha=true,beta=false,allocator=TensorOperations.DefaultAllocator()) =
    _execute_transform_table!(t_dst,t_src,bulk,alpha,beta,allocator)


# tensorcontract
function create_mediated_planarcontract!(C::SymbolicTensorMap, A::SymbolicTensorMap, pA, B::SymbolicTensorMap, pB, pC, α=1, β=0 , backend=nothing)
    S = spacetype(A.structure)

    codA, domA = codomainind(A), domainind(A)
    codB, domB = codomainind(B), domainind(B)
    # like TensorKit.planarcontract!: rotate to cyclic partitions, C = transpose(A′*B′, pC′)
    (oindA, cindA), (cindB, oindB), pC′ = TensorKit.planar_contract_indices(A.structure, pA, B.structure, pB, pC)

    #A′ = transpose(A, (oindA, cindA))
    sp_dst_A =  ProductSpace{S,length(oindA)}(map(n -> A.structure[n], oindA)) ← ProductSpace{S,length(cindA)}(map(n -> dual(A.structure[n]), cindA))
    fast_init_A = fast_init(codomain(sp_dst_A),domain(sp_dst_A),storagetype(ttype(A)))
    tbl_A = generate_transpose_table(scalartype(ttype(A)),A.structure,sp_dst_A,oindA,cindA)
    inplace_A = (oindA == codA && cindA == domA)

    #B′ = transpose(B, (cindB, oindB))
    sp_dst_B =  ProductSpace{S,length(cindB)}(map(n -> B.structure[n], cindB)) ← ProductSpace{S,length(oindB)}(map(n -> dual(B.structure[n]), oindB))
    fast_init_B = fast_init(codomain(sp_dst_B),domain(sp_dst_B),storagetype(ttype(B)))
    tbl_B = generate_transpose_table(scalartype(ttype(B)),B.structure,sp_dst_B,cindB,oindB)
    inplace_B =  (cindB == codB && oindB == domB)

    # a non-trivial pC′ requires an intermediate A′*B′
    direct_C = TensorKit._isdirectoutput(pC′, length(oindA))
    sp_AB = codomain(sp_dst_A) ← domain(sp_dst_B)
    fast_init_AB = fast_init(codomain(sp_AB),domain(sp_AB),storagetype(ttype(C)))
    tbl_C = direct_C ? nothing : generate_transpose_table(scalartype(ttype(C)),sp_AB,C.structure,pC′[1],pC′[2])

    (C,(fast_init_A,tbl_A,fast_init_B,tbl_B,inplace_A,inplace_B,direct_C,fast_init_AB,tbl_C))
end

function mediated_planarcontract!(fst,mediator,C, A, pA::Index2Tuple, B, pB::Index2Tuple, pC::Index2Tuple, α=1, β=0 , backend=nothing)
    (fast_init_A,tbl_A,fast_init_B,tbl_B,inplace_A,inplace_B,direct_C,fast_init_AB,tbl_C) = mediator

    if inplace_A
        Ap = A
    else
        Ap = fast_init_A(fst.allocator,Val(true))
        execute_transpose_table!(Ap,A,tbl_A,true,false,fst.allocator)
    end

    if inplace_B
        Bp = B
    else
        Bp = fast_init_B(fst.allocator,Val(true))
        execute_transpose_table!(Bp,B,tbl_B,true,false,fst.allocator)
    end

    if direct_C
        mul!(C,Ap,Bp,α,β)
    else
        AB = mul!(fast_init_AB(fst.allocator,Val(true)),Ap,Bp)
        execute_transpose_table!(C,AB,tbl_C,α,β,fst.allocator)
        tensorfree!(AB, fst.allocator)
    end
    !inplace_A && tensorfree!(Ap, fst.allocator)
    !inplace_B && tensorfree!(Bp, fst.allocator)

    C
end



# tensoradd
function create_mediated_planaradd!(C, A, pC, α, β , backend=nothing)
    tbl_transpose = generate_transpose_table(scalartype(ttype(C)),A.structure,C.structure,pC[1],pC[2])
    (C,(tbl_transpose,))
end

function mediated_planaradd!(fst,mediator,C , A,pC, α, β , backend=nothing)
    (tbl_transpose,) = mediator
    execute_transpose_table!(C,A,tbl_transpose,α,β,fst.allocator)
   
    C
end

# tensortrace 
function create_mediated_planartrace!(C, pC, A, pA, conjA, α=1, β=0 , backend=nothing)
    @show "not yet planartrace"
    (C,Nothing)
end

function mediated_planartrace!(fst,mediator,args...)
    TensorKit.planartrace!(args...)
end
