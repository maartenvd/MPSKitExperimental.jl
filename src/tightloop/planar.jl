# transpos 
function generate_transpose_table(elt,sp_src,sp_dst, p1::IndexTuple{N₁},p2::IndexTuple{N₂}) where {N₁,N₂}
    
    blocksectoriterator_src = blocksectors(sp_src);
    rowr_src, rowdims = TensorKit._buildblockstructure(codomain(sp_src), blocksectoriterator_src)
    colr_src, coldims = TensorKit._buildblockstructure(domain(sp_src), blocksectoriterator_src)

    blocksectoriterator_dst = blocksectors(sp_dst);
    rowr_dst, rowdims = TensorKit._buildblockstructure(codomain(sp_dst), blocksectoriterator_dst)
    colr_dst, coldims = TensorKit._buildblockstructure(domain(sp_dst), blocksectoriterator_dst)

    ftreemap = (f1, f2)->transpose(f1, f2, p1, p2);
    I = eltype(rowr_src.keys);

    N = length(p1)+length(p2);
    table = Tuple{elt,Int,UnitRange{Int},UnitRange{Int},NTuple{N,Int},Int,UnitRange{Int},UnitRange{Int},NTuple{N,Int}}[];
    @inbounds for (i_src,(s_src,f1_list_src)) in enumerate(rowr_src)
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

function execute_transpose_table!(t_dst,t_src,bulk,alpha=true,beta=false)
    
    (table,p1,p2) = bulk
    rmul!(t_dst,beta);

    for (α,s_src,r_src,c_src,d_src,s_dst,r_dst,c_dst,d_dst) in table
        if first(p1) == 1
            axpy!(α*alpha,(@view t_src.data.values[s_src][r_src,c_src]),(@view t_dst.data.values[s_dst][r_dst,c_dst]))
        else
            view_dst = sreshape(StridedView(t_dst.data.values[s_dst])[r_dst,c_dst],d_dst)
            view_src = sreshape(StridedView(t_src.data.values[s_src])[r_src,c_src],d_src);
            axpy!(α*alpha,permutedims(view_src,(p1...,p2...)), view_dst);
        end
    end

    t_dst
end


# tensorcontract
function create_mediated_planarcontract!(C::SymbolicTensorMap, pC, A::SymbolicTensorMap, pA, B::SymbolicTensorMap, pB, α=1, β=0 , backend=nothing)
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

function mediated_planarcontract!(fst,mediator,C, pC, A, pA, B, pB, α=1, β=0 , backend=nothing)
    (fast_init_A,tbl_A,fast_init_B,tbl_B,inplace_A,inplace_B) = mediator

    if inplace_A
        #@show "inplace A"
        Ap = A
    else
        #@show "transpose A"
        Ap = fast_init_A(fst.allocator,true)
        execute_transpose_table!(Ap,A,tbl_A)    
    end

    if inplace_B
        #@show "inplace B"
        Bp = B       
    else
        #@show "transpose B"
        Bp = fast_init_B(fst.allocator,true)
        execute_transpose_table!(Bp,B,tbl_B)
    end

    mul!(C,Ap,Bp,α,β)

    !inplace_A && tensorfree!(Ap, fst.allocator)
    !inplace_B && tensorfree!(Bp, fst.allocator)
   
    C    
end



# tensoradd
function create_mediated_planaradd!(C, pC, A, α, β , backend=nothing)
    tbl_transpose = generate_transpose_table(scalartype(ttype(C)),A.structure,C.structure,pC[1],pC[2])
    (C,(tbl_transpose,))
end

function mediated_planaradd!(fst,mediator,C, pC, A, α, β , backend=nothing)
    (tbl_transpose,) = mediator
    execute_transpose_table!(C,A,tbl_transpose,α,β)
   
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
