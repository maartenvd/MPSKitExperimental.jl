# transpos
function generate_transpose_table(elt,sp_src,sp_dst, p1::IndexTuple{N₁},p2::IndexTuple{N₂}) where {N₁,N₂}
    transformer = TensorKit.treetransposer(sp_dst, sp_src, (p1,p2))
    (transformer,p1,p2)
end

function execute_transpose_table!(t_dst,t_src,bulk::Tuple{TensorKit.TrivialTreeTransformer,Any,Any},alpha=true,beta=false,allocator=TensorOperations.DefaultAllocator())
    (transformer,p1,p2) = bulk
    TensorOperations.tensoradd!(t_dst[],t_src[],(p1,p2),false,alpha,beta,
        TensorOperations.DefaultBackend(),allocator)
    t_dst
end

function execute_transpose_table!(t_dst,t_src,bulk,alpha=true,beta=false,allocator=TensorOperations.DefaultAllocator())
    (transformer,p1,p2) = bulk
    TensorKit.add_transform_kernel!(t_dst.data,t_src.data,(p1,p2),transformer,alpha,beta,
        TensorOperations.DefaultBackend(),allocator,TensorKit.OhMyThreads.SerialScheduler())
    t_dst
end


# tensorcontract
function create_mediated_planarcontract!(C::SymbolicTensorMap, A::SymbolicTensorMap, pA, B::SymbolicTensorMap, pB, pC, α=1, β=0 , backend=nothing)
    S = spacetype(A.structure)

    codA, domA = codomainind(A), domainind(A)
    codB, domB = codomainind(B), domainind(B)
    oindA, cindA = pA
    cindB, oindB = pB
    oindA, cindA, oindB, cindB = TensorKit.reorder_indices(codA, domA, codB, domB, oindA, cindA,
                                                 oindB, cindB, pC...)

    #A′ = permute(A, (oindA, cindA); copy=copyA)
    sp_dst_A =  ProductSpace{S,length(oindA)}(map(n -> A.structure[n], oindA)) ← ProductSpace{S,length(cindA)}(map(n -> dual(A.structure[n]), cindA))
    fast_init_A = fast_init(codomain(sp_dst_A),domain(sp_dst_A),storagetype(ttype(A)))
    tbl_A = generate_transpose_table(scalartype(ttype(A)),A.structure,sp_dst_A,oindA,cindA)
    inplace_A = (oindA == codA && cindA == domA)

    #B′ = permute(B, (cindB, oindB))
    sp_dst_B =  ProductSpace{S,length(cindB)}(map(n -> B.structure[n], cindB)) ← ProductSpace{S,length(oindB)}(map(n -> dual(B.structure[n]), oindB))
    fast_init_B = fast_init(codomain(sp_dst_B),domain(sp_dst_B),storagetype(ttype(B)))
    tbl_B = generate_transpose_table(scalartype(ttype(B)),B.structure,sp_dst_B,cindB,oindB)
    inplace_B =  (cindB == codB && oindB == domB)
    
    (C,(fast_init_A,tbl_A,fast_init_B,tbl_B,inplace_A,inplace_B))
end

function mediated_planarcontract!(fst,mediator,C, A, pA::Index2Tuple, B, pB::Index2Tuple, pC::Index2Tuple, α=1, β=0 , backend=nothing)
    (fast_init_A,tbl_A,fast_init_B,tbl_B,inplace_A,inplace_B) = mediator

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

    mul!(C,Ap,Bp,α,β)
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
