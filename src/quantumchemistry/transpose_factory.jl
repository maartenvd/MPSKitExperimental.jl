#=
transposition/permutation require
    - allocation of new data -> need rowr/colr/codom/dom/storagetype
    - a list of (scalar,s_src,rowslice_src,colslice_src,s_dst,rowslice_dst,colslice_dst)
=#


struct TransposeFact{A<:DelayedFact,N,E}
    delayed::A
    pdata::NTuple{N,Int}
    table::Vector{Tuple{E,Int,UnitRange{Int},UnitRange{Int},NTuple{N,Int},Int,UnitRange{Int},UnitRange{Int},NTuple{N,Int}}};
end

function TransposeFactType(S,storage,N₁,N₂)
    A = DelayedFactType(S,storage,N₁,N₂)
    TransposeFact{A,N₁+N₂,eltype(storage)}
end

function TransposeFact(orig::DelayedFact, p1::Tuple,p2::Tuple)
    homsp = orig.cod ← orig.dom;
    S = spacetype(homsp);
    cod = ProductSpace{S}(map(n->homsp[n], p1))
    dom = ProductSpace{S}(map(n->dual(homsp[n]), p2))
    storage = orig.storage;
    delayed = DelayedFact(cod ← dom,storage);

    rowr_src = orig.rowr;
    colr_src = orig.colr;
    rowr_dst = delayed.rowr;
    colr_dst = delayed.colr;
    
    # generate the actual table
    table = generate_table(eltype(storage),homsp,rowr_src,colr_src,cod←dom,rowr_dst,colr_dst,p1,p2);

    
    TransposeFact(delayed,(p1...,p2...),table);

end
function TransposeFact(homsp,storage,p1,p2)
    S = spacetype(homsp);

    cod = ProductSpace{S}(map(n->homsp[n], p1))
    dom = ProductSpace{S}(map(n->dual(homsp[n]), p2))

    (rowr_src,colr_src,rowdims_src,coldims_src) = calc_rowr_color(sectortype(homsp),codomain(homsp),domain(homsp));
   
    delayed = DelayedFact(cod ← dom,storage);
    rowr_dst = delayed.rowr;
    colr_dst = delayed.colr;
    
    
    # generate the actual table
    table = generate_table(eltype(storage),homsp,rowr_src,colr_src,cod←dom,rowr_dst,colr_dst,p1,p2);

    
    TransposeFact(delayed,(p1...,p2...),table);
end

function (fact::TransposeFact{A,E})(t::TensorMap) where {A,E}
    
    
    # allocate the output tensormap
    tdst = fact.delayed()
    rmul!(tdst,false);

    # iterate over elements in the table
    @inbounds for (α,s_src,r_src,c_src,d_src,s_dst,r_dst,c_dst,d_dst) in fact.table
        if first(fact.pdata) == 1
            axpy!(α,(@view t.data.values[s_src][r_src,c_src]),(@view tdst.data.values[s_dst][r_dst,c_dst]))
        else
            view_dst = sreshape(StridedView(tdst.data.values[s_dst])[r_dst,c_dst],d_dst)
            view_src = sreshape(StridedView(t.data.values[s_src])[r_src,c_src],d_src);

            axpy!(α,permutedims(view_src,fact.pdata), view_dst);
        end
        
    end

    tdst
end

function generate_table(elt,sp_src,rowr_src,colr_src,sp_dst,rowr_dst,colr_dst,p1,p2)
    ftreemap = (f1, f2)->transpose(f1, f2, p1, p2);
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

                r_dst = rowr_dst.values[i_dst][f1_dst];
                c_dst = colr_dst.values[i_dst][f2_dst];


                push!(table,(α,i_src,r_src,c_src,d_src,i_dst,r_dst,c_dst,d_dst));
            end
        end
    end

    table
end
