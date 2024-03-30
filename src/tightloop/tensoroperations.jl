# permute 
function generate_permute_table(elt,sp_src,sp_dst, p1::IndexTuple{N₁},p2::IndexTuple{N₂}) where {N₁,N₂}
    
    blocksectoriterator_src = blocksectors(sp_src);
    rowr_src, rowdims = TensorKit._buildblockstructure(codomain(sp_src), blocksectoriterator_src)
    colr_src, coldims = TensorKit._buildblockstructure(domain(sp_src), blocksectoriterator_src)

    blocksectoriterator_dst = blocksectors(sp_dst);
    rowr_dst, rowdims = TensorKit._buildblockstructure(codomain(sp_dst), blocksectoriterator_dst)
    colr_dst, coldims = TensorKit._buildblockstructure(domain(sp_dst), blocksectoriterator_dst)

    ftreemap = (f1, f2)->permute(f1, f2, p1, p2);
    I = eltype(rowr_src.keys);

    N = length(p1)+length(p2);
    table = Tuple{elt,Int,UnitRange{Int},UnitRange{Int},NTuple{N,Int},Int,UnitRange{Int},UnitRange{Int},NTuple{N,Int}}[];
    for (i_src,(s_src,f1_list_src)) in enumerate(rowr_src)
        f2_list_src = colr_src[s_src];

        for (f1_src,r_src) in f1_list_src, (f2_src,c_src) in f2_list_src
            d_src = (dims(codomain(sp_src), f1_src.uncoupled)..., dims(domain(sp_src), f2_src.uncoupled)...)
            for ((f1_dst,f2_dst),α) in ftreemap(f1_src,f2_src)
                
                d_dst = (dims(codomain(sp_dst), f1_dst.uncoupled)..., dims(domain(sp_dst), f2_dst.uncoupled)...)

                s_dst = f1_dst.coupled;
                

                i_dst = searchsortedfirst(rowr_dst.keys,s_dst);
                (i_dst > length(rowr_dst.keys) || rowr_dst.keys[i_dst] != s_dst) && continue
                r_dst = rowr_dst.values[i_dst][f1_dst];
                c_dst = colr_dst.values[i_dst][f2_dst];


                push!(table,(α,i_src,r_src,c_src,d_src,i_dst,r_dst,c_dst,d_dst));
            end
        end
    end

    (table,p1,p2)
end

function execute_permute_table!(t_dst,t_src,bulk,beta=false)
    (table,p1,p2) = bulk
    rmul!(t_dst,beta);

    @inbounds for (α,s_src,r_src,c_src,d_src,s_dst,r_dst,c_dst,d_dst) in table
        view_dst = sreshape(StridedView(t_dst.data.values[s_dst])[r_dst,c_dst],d_dst)
        view_src = sreshape(StridedView(t_src.data.values[s_src])[r_src,c_src],d_src);
        
        #TensorOperations.tensoradd!(view_dst,(p1,p2),view_src,:N,α,true)
        axpy!(α,permutedims(view_src,(p1...,p2...)), view_dst);
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
function create_mediated_tensoralloc_add(TC, pC::Index2Tuple{N₁,N₂}, A::SymbolicTensorMap, conjA, istemp=false, backend::TensorOperations.Backend...)  where {N₁,N₂}

    S = spacetype(ttype(A))
    cod = ProductSpace{S,N₁}(broadcast(p->TensorOperations.flag2op(conjA)(A.structure[p]),pC[1]))
    dom = ProductSpace{S,N₂}(broadcast(p->dual(TensorOperations.flag2op(conjA)(A.structure[p])),pC[2]))
    stortype = TensorKit.similarstoragetype(ttype(A),TC)
    C = SymbolicTensorMap(tensormaptype(S,N₁, N₂, stortype),dom → cod)

    (C,fast_init(cod,dom,stortype))
end

function mediated_tensoralloc_add(fst,mediator,TC, pC::Index2Tuple{N₁,N₂}, A, conjA, istemp=false, backend::TensorOperations.Backend...)  where {N₁,N₂}
    mediator(fst.allocator,istemp)
end

# tensortrace 
function create_mediated_tensortrace!(C, pC, A, pA, conjA, α=1, β=0 , backend=nothing)
    (C,Nothing)
end

function mediated_tensortrace!(fst,mediator,args...)
    TensorOperations.tensortrace!(args...)
end


# tensorcontract
function create_mediated_tensorcontract!(C::SymbolicTensorMap, pC, A::SymbolicTensorMap, pA, conjA, B::SymbolicTensorMap, pB, conjB, α=1, β=0 , backend=nothing)
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
    A_structure = conjA == :N ? A.structure : conj(codomain(A.structure))←conj(domain(A.structure))
    sp_dst_A =  ProductSpace{S,length(pA[1])}(map(n -> A_structure[n], pA[1])) ← ProductSpace{S,length(pA[2])}(map(n -> dual(A_structure[n]), pA[2]))
    fast_init_A = fast_init(codomain(sp_dst_A),domain(sp_dst_A),storagetype(ttype(A)))
    tbl_A = generate_permute_table(scalartype(ttype(A)),A_structure,sp_dst_A,pA[1],pA[2])

    #B′ = permute(B, (cindB, oindB))
    B_structure = conjB == :N ? B.structure : conj(codomain(B.structure))←conj(domain(B.structure))
    sp_dst_B =  ProductSpace{S,length(pB[1])}(map(n -> B_structure[n], pB[1])) ← ProductSpace{S,length(pB[2])}(map(n -> dual(B_structure[n]), pB[2]))
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

function mediated_tensorcontract!(fst,mediator,C, pC, A, pA, conjA, B, pB, conjB, α=1, β=0 , backend=nothing)
    (fast_init_A,tbl_A,fast_init_B,tbl_B,fast_init_C′,tbl_C′) = mediator

    cleanup_A = false
    tot_pA = (pA[1]...,pA[2]...)
    if tot_pA == ntuple(identity,length(tot_pA)) && length(pA[1]) == length(codomain(A)) && length(pA[2]) == length(domain(A))
        Ap = A
    else
        cleanup_A = true
        Ap = fast_init_A(fst.allocator,true)
        execute_permute_table!(Ap,A,tbl_A)    
    end

    cleanup_B = false
    tot_pB = (pB[1]...,pB[2]...)
    if tot_pB == ntuple(identity,length(tot_pB)) && length(pB[1]) == length(codomain(B)) && length(pB[2]) == length(domain(B))
        Bp = B
                
    else
        cleanup_B = true
        Bp = fast_init_B(fst.allocator,true)
        execute_permute_table!(Bp,B,tbl_B)
    end

    cleanup_C = false
    if pC[1] == ntuple(n -> n, length(codomain(Ap))) && pC[2] == ntuple(n->n+length(codomain(Ap)),length(domain(Bp)))
        mul!(C,Ap,Bp,α,β)
    else
        cleanup_C = true
        C′ = mul!(fast_init_C′(fst.allocator,true),Ap,Bp,α)
        execute_permute_table!(C,C′,tbl_C′,β)
    end

    cleanup_A && tensorfree!(Ap, fst.allocator)
    cleanup_B && tensorfree!(Bp, fst.allocator)
    cleanup_C && tensorfree!(C′, fst.allocator)

    C    
end


# tensoralloc_contract
function create_mediated_tensoralloc_contract(TC, pC::Index2Tuple{N₁,N₂}, A::SymbolicTensorMap, pA, conjA, B::SymbolicTensorMap, pB, conjB, istemp=false, backend::TensorOperations.Backend...)  where {N₁,N₂}
    spaces1 = [TensorOperations.flag2op(conjA)(A.structure[p]) for p in pA[1]]
    spaces2 = [TensorOperations.flag2op(conjB)(B.structure[p]) for p in pB[2]]
    spaces = (spaces1..., spaces2...)

    S = spacetype(ttype(A))
    cod = ProductSpace{S,N₁}(getindex.(Ref(spaces), pC[1]))
    dom = ProductSpace{S,N₂}(dual.(getindex.(Ref(spaces), pC[2])))
    stortype = TensorKit.similarstoragetype(ttype(A),TC)
    C = SymbolicTensorMap(tensormaptype(S,N₁, N₂, stortype),dom → cod)

    (C,fast_init(cod,dom,stortype)) 
end

function mediated_tensoralloc_contract(fst,mediator,TC, pC::Index2Tuple{N₁,N₂}, A, pA, conjA, B, pB, conjB, istemp=false, backend::TensorOperations.Backend...)  where {N₁,N₂}
    mediator(fst.allocator,istemp)
end