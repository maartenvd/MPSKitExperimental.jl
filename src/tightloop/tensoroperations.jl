# transpos 
function generate_permute_table(elt,sp_src,sp_dst, p1::IndexTuple{N₁},p2::IndexTuple{N₂}) where {N₁,N₂}
    str_src = TensorKit.fusionblockstructure(sp_src)
    str_dst = TensorKit.fusionblockstructure(sp_dst)


    N = length(p1)+length(p2);
    table = Tuple{elt,Tuple{NTuple{N,Int},NTuple{N,Int},Int},Tuple{NTuple{N,Int},NTuple{N,Int},Int}}[];
    for (i,(f1,f2)) in enumerate(str_src.fusiontreelist)
        cur_str_src = str_src.fusiontreestructure[i]
        for ((f3,f4),coeff) in permute(f1, f2, p1, p2)
            cur_str_dst = str_dst.fusiontreestructure[str_dst.fusiontreeindices[(f3,f4)]]
            #StridedView(t.data, sz, str, offset)
            push!(table,(coeff,cur_str_src,cur_str_dst))
        end
    end
    
    (table,p1,p2)
end

function execute_permute_table!(t_dst,t_src,bulk,alpha=true,beta=false)
    
    (table,p1,p2) = bulk
    rmul!(t_dst,beta);
    
    for (α, cur_str_src,cur_str_dst) in table
        view_src = StridedView(t_src.data, cur_str_src...)
        view_dst = StridedView(t_dst.data, cur_str_dst...)
        axpy!(α*alpha,permutedims(view_src,(p1...,p2...)), view_dst)
    end

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

Base.conj(P::ProductSpace) = ProductSpace(map(conj, P.spaces))

# tensorcontract
function create_mediated_tensorcontract!(C::SymbolicTensorMap, A::SymbolicTensorMap, pA, conjA, B::SymbolicTensorMap, pB, conjB, pC,  α=1, β=0 , b1=nothing, b2=nothing)
    S = spacetype(A.structure)
    if !(BraidingStyle(sectortype(S)) isa SymmetricBraiding)
        throw(SectorMismatch("only tensors with symmetric braiding rules can be contracted; try `@planar` instead"))
    end
    #=
    copyA = false
    if BraidingStyle(sectortype(S)) isa Fermionic
        for i in cindA
            if !isdual(space(A, i))
                copyA = true
            end
        end
    end
    =#

    #A′ = permute(A, (oindA, cindA); copy=copyA)
    A_structure = !conjA ? A.structure : conj(codomain(A.structure)) ← conj(domain(A.structure))
    sp_dst_A =  ProductSpace{S,length(pA[1])}(map(n -> A_structure[n], pA[1])) ← ProductSpace{S,length(pA[2])}(map(n -> conj(A_structure[n]), pA[2]))
    fast_init_A = fast_init(codomain(sp_dst_A),domain(sp_dst_A),storagetype(ttype(A)))
    tbl_A = generate_permute_table(scalartype(ttype(A)),A_structure,sp_dst_A,pA[1],pA[2])

    #B′ = permute(B, (cindB, oindB))
    B_structure = !conjB  ? B.structure : conj(codomain(B.structure)) ← conj(domain(B.structure))
    sp_dst_B =  ProductSpace{S,length(pB[1])}(map(n -> B_structure[n], pB[1])) ← ProductSpace{S,length(pB[2])}(map(n -> conj(B_structure[n]), pB[2]))
    fast_init_B = fast_init(codomain(sp_dst_B),domain(sp_dst_B),storagetype(ttype(B)))
    tbl_B = generate_permute_table(scalartype(ttype(B)),B_structure,sp_dst_B,pB[1],pB[2])
    
    #=
    if BraidingStyle(sectortype(S)) isa Fermionic
        for i in domainind(A′)
            if !isdual(space(A′, i))
                A′ = twist!(A′, i)
            end
        end
    end
    =#
    #=
    ipC = TupleTools.invperm((pC[1]..., pC[2]...))
    oindAinC = TupleTools.getindices(ipC, ntuple(n -> n, N₁))
    oindBinC = TupleTools.getindices(ipC, ntuple(n -> n + N₁, N₂))
    if has_shared_permute(C, (oindAinC, oindBinC))
        C′ = permute(C, (oindAinC, oindBinC))
        mul!(C′, A′, B′, α, β)
    else
        C′ = A′ * B′
        add_permute!(C, C′, (p₁, p₂), α, β)
    end
    return C
    =#

    
    fast_init_C′ = fast_init(codomain(sp_dst_A),domain(sp_dst_B),storagetype(ttype(C)));
    tbl_C′ = generate_permute_table(scalartype(ttype(C)),codomain(sp_dst_A)←domain(sp_dst_B),C.structure,pC[1],pC[2])

    (C,(fast_init_A,tbl_A,fast_init_B,tbl_B,fast_init_C′,tbl_C′))
end

function mediated_tensorcontract!(fst,mediator,C, A, pA, conjA, B, pB, conjB, pC, α=1, β=0 , b1=nothing, b2=nothing)
    (fast_init_A,tbl_A,fast_init_B,tbl_B,fast_init_C′,tbl_C′) = mediator

    
    cleanup_A = false
    tot_pA = (pA[1]...,pA[2]...)
    if tot_pA == ntuple(identity,length(tot_pA)) && length(pA[1]) == length(codomain(A)) && length(pA[2]) == length(domain(A))
        Ap = A
    else
        cleanup_A = true
        Ap = fast_init_A(fst.allocator,Val(true))
        execute_permute_table!(Ap,A,tbl_A)    
    end
  

    cleanup_B = false
    tot_pB = (pB[1]...,pB[2]...)
    if tot_pB == ntuple(identity,length(tot_pB)) && length(pB[1]) == length(codomain(B)) && length(pB[2]) == length(domain(B))
        Bp = B
                
    else
        cleanup_B = true
        Bp = fast_init_B(fst.allocator,Val(true))
        execute_permute_table!(Bp,B,tbl_B)
    end

    cleanup_C = false
    if pC[1] == ntuple(n -> n, length(codomain(Ap))) && pC[2] == ntuple(n->n+length(codomain(Ap)),length(domain(Bp)))
        mul!(C,Ap,Bp,α,β)
    else
        cleanup_C = true
        C′ = mul!(fast_init_C′(fst.allocator,Val(true)),Ap,Bp,α) 
        execute_permute_table!(C,C′,tbl_C′,true,β)
    end

    cleanup_A && tensorfree!(Ap, fst.allocator)
    cleanup_B && tensorfree!(Bp, fst.allocator)
    cleanup_C && tensorfree!(C′, fst.allocator)

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