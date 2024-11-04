struct fused_∂∂AC{A}
    blocks::A
end

function MPSKit.∂∂AC(pos::Int,mps,ham::FusedMPOHamiltonian,cache)
    opp = ham[pos];
    le = leftenv(cache,pos,mps);
    re = rightenv(cache,pos,mps);

    process_blocks = Map() do (lmask,lblock,e,rblock,rmask)
        cl = le[lmask];
        cr = re[rmask];

        
        l = rmul!(fast_copy(cl[1]),lblock[1])
        for i in 2:length(cl)
            l = fast_axpy!(lblock[i],cl[i],l);
        end

        r = rmul!(fast_copy(cr[1]),rblock[1])
        for i in 2:length(rblock)
            r = fast_axpy!(rblock[i],cr[i],r);
        end        
        
        (l,e,r)
    end


    blocks = tcollect(process_blocks,opp.blocks)

    filter!(blocks) do (l,e,r)
        !(norm(l)<1e-12 || norm(r)<1e-12)
    end
    
    fused_∂∂AC(blocks)
end

function (h::fused_∂∂AC)(x)
    @floop for (l,e,r) in h.blocks
        @init t = similar(x)
        
        @planar allocator=malloc() t[-1 -2;-3] = l[-1 5; 4] * x[4 2; 1] * e[5 -2; 2 3] * r[1 3; -3]

        @reduce() do (toret = zero(x); t)
            fast_axpy!(true,t,toret);
            toret
        end
    end
    
    toret
end

Base.:*(a::fused_∂∂AC,v) = a(v)

# ugly - inconsistent with MPOHamiltonian
MPSKit.expectation_value(st::FiniteMPS,th::FusedMPOHamiltonian,envs = environments(st,th)) =
    dot(st.AC[1],MPSKit.∂∂AC(1,st,th,envs)(st.AC[1]))/dot(st.AC[1],st.AC[1])


struct fused_∂∂AC2{A}
    table::A
    buffersize::Int
end

function rowr_colr_from_fusionblockstructure(structure::TensorKit.FusionBlockStructure{I,N,F₁,F₂}) where {I,N,F₁,F₂}
    S = sectortype(F₁)
    rowr = Dict{S,Dict{F₁,UnitRange{Int}}}()
    colr = Dict{S,Dict{F₂,UnitRange{Int}}}()
    N1 = length(F₁)
    N2 = length(F₂)
    for ((f1,f2),(sz,st,o)) in zip(structure.fusiontreelist,structure.fusiontreestructure)
        (block_sz,block_range) = structure.blockstructure[f1.coupled]
        block_range_start = block_range[1]
        
        #(subsz, substr, totaloffset)

        # reshape back to codomain/domain:
        i = mod(o+1-block_range_start,block_sz[1])+1
        j = (o+1-block_range_start)÷block_sz[1]+1

        irange = i:i+prod(sz[1:N1])-1
        jrange = j:j+prod(sz[N1+1:N1+N2])-1

        if !(f1.coupled in keys(rowr))
            rowr[f1.coupled] = Dict{F₁,UnitRange{Int}}()
            colr[f1.coupled] = Dict{F₁,UnitRange{Int}}()
        end
        if f1 in keys(rowr[f1.coupled])
            @assert rowr[f1.coupled][f1] == irange
        else
            rowr[f1.coupled][f1] = irange
        end

        if f2 in keys(colr[f1.coupled])
            @assert colr[f1.coupled][f2] == jrange
        else
            colr[f1.coupled][f2] = jrange
        end
    end

    return (rowr,colr)
end

function _untrip_row(rowr::Dict{S,Dict{F₁,UnitRange{Int}}}) where {S,F₁}
    out = Dict{S,Dict{S,UnitRange{Int}}}() #this will probably fail for multiple fusion, as innerlines aren't sectors
    for (k,v) in rowr
        if !(k in keys(out))
            out[k] = Dict{S,UnitRange{Int}}()
        end

        for (f,range) in v
            q = f.uncoupled[end]
            if !(q in keys(out[k]))
                out[k][q] = range
            else
                out[k][q] = min(range.start,out[k][q].start):max(range.stop,out[k][q].stop)
            end
        end
    end
    return out
end


function _untrip_col(colr::Dict{S,Dict{F₁,UnitRange{Int}}}) where {S,F₁}
    out = Dict{S,Dict{Tuple{S,S,S},UnitRange{Int}}}() #this will probably fail for multiple fusion, as innerlines aren't sectors
    for (k,v) in colr
        if !(k in keys(out))
            out[k] = Dict{Tuple{S,S,S},UnitRange{Int}}()
        end

        for (f,range) in v
            q = (f.uncoupled[end],f.innerlines[end],f.uncoupled[end-1])
            if !(q in keys(out[k]))
                out[k][q] = range
            else
                out[k][q] = min(range.start,out[k][q].start):max(range.stop,out[k][q].stop)
            end
        end
    end
    return out
end


function _leftblock(opp1::FusedSparseBlock{E,O,Sp},le) where {E,O,Sp}
   
    blocked_left_blocks = map(opp1.blocks) do (lmask,lblock,e,rblock,rmask)
        cl = le[lmask];
        
        l = rmul!(copy(cl[1]),lblock[1])
        for i in 2:length(cl)
            l = axpy!(lblock[i],cl[i],l);
        end

        @planar allocator=malloc cle[-1 -2;-3 -4 -5] := l[-1 1;-3]*e[1 -2;-4 -5]
        
        (rowr,colr) = rowr_colr_from_fusionblockstructure(TensorKit.fusionblockstructure(cle.space))
        sparsified = Dict{Tuple{sectortype(l),sectortype(l),sectortype(l),sectortype(l),sectortype(l)},Matrix{eltype(l)}}()

        untr_col = _untrip_col(colr)
        untr_row = _untrip_row(rowr)

        for (q2,b) in blocks(cle)
            for (q1,rowrange) in untr_row[q2], ((q3,q4,q5),colrange) in untr_col[q2]
                norm(b[rowrange,colrange],Inf) < 1e-12 && continue
                
                sparsified[(q1,q2,q3,q4,q5)] = copy(b[rowrange,colrange])
            end
        end

        (sparsified,space(e,4),rblock,rmask)
    end

    
    filter!(blocked_left_blocks) do (l,sp,rblock,rmask)
        !isempty(l) && !isempty(rblock)
    end
    
    return blocked_left_blocks
end

function _rightblock(opp2::FusedSparseBlock{E,O,Sp},re) where {E,O,Sp}

  
    blocked_right_blocks = map(opp2.blocks) do (lmask,lblock,e,rblock,rmask)
        cr = re[rmask];

        r = rmul!(copy(cr[1]),rblock[1])
        for i in 2:length(rblock)
            r = axpy!(rblock[i],cr[i],r)
        end

        @planar allocator=malloc cre[-1 -2 -3;-4 -5] := r[-1 1;-4]*e[-3 -5;-2 1]
        
        (rowr,colr) = rowr_colr_from_fusionblockstructure(TensorKit.fusionblockstructure(cre.space))
        sparsified = Dict{Tuple{sectortype(r),sectortype(r),sectortype(r),sectortype(r),sectortype(r)},Matrix{eltype(r)}}()

        untr_col = _untrip_row(colr)
        untr_row = _untrip_col(rowr)

        for (q2,b) in blocks(cre)
            for (q1,colrange) in untr_col[q2], ((q3,q4,q6),rowrange) in untr_row[q2]
                norm(b[rowrange,colrange],Inf) < 1e-12 && continue
                
                sparsified[(q6,q4,q3,q2,q1)] = copy(b[rowrange,colrange])
            end
        end

        (lmask,lblock,space(e,1),sparsified)
    end

    
    filter!(blocked_right_blocks) do (lmask,lblock,sp,r)
        !isempty(lblock) && !isempty(r)
    end
    
    return blocked_right_blocks
end

function MPSKit.∂∂AC2(pos::Int,mps,ham::FusedMPOHamiltonian{E,O,Sp},cache) where {E,O,Sp}
    opp1 = ham[pos];
    opp2 = ham[pos+1];

    le = leftenv(cache,pos,mps);
    re = rightenv(cache,pos+1,mps);
    p1 = opp1.pspace;
    p2 = opp2.pspace;
    v1 = left_virtualspace(mps,pos-1);
    v2 = right_virtualspace(mps,pos+1);
    ac2_structure = v1*p1 ← v2*(p2)'
    S = sectortype(ac2_structure)

    ac2_blockstructure = TensorKit.fusionblockstructure(ac2_structure)
    (rowr_ac2,colr_ac2) = rowr_colr_from_fusionblockstructure(ac2_blockstructure)
    left_ac2_untrp = _untrip_row(rowr_ac2)
    right_ac2_untrp = _untrip_row(colr_ac2)
    
    
    blocked_left_blocks = _leftblock(opp1,le)
    blocked_right_blocks = _rightblock(opp2,re)

    left_group = Dict{typeof(opp1.pspace),Vector{Int}}();
    right_group = Dict{typeof(opp1.pspace),Vector{Int}}();

    for (i,(l,sp,rblock,rmask)) in enumerate(blocked_left_blocks)
        left_group[sp'] = [i;get(left_group,sp',Int[])]
    end
    
    for (i,(lmask,lblock,sp, r)) in enumerate(blocked_right_blocks)
        right_group[sp] = [i;get(right_group,sp,Int[])]
    end
    common_keys = collect(intersect(keys(left_group),keys(right_group)))
    d_matrices = Dict(map(k-> k=> fill(zero(E),length(left_group[k]),length(right_group[k])),common_keys))

    for (k,v) in left_group, (il,lv) in enumerate(v), (ir,rv) in enumerate(get(right_group,k,Int[]))
        (l,e1,rblock,rmask) = blocked_left_blocks[lv]
        (lmask,lblock,e2,r) = blocked_right_blocks[rv]

        d_matrices[k][il,ir] = sum(rblock[lmask[rmask]].*lblock[rmask[lmask]]);
    end
    pairs = map(collect(keys(d_matrices))) do k
        d = d_matrices[k]
        
        (U_s,R_s) = qr(d);
        U = Matrix(U_s);
        R = Matrix(R_s);

        flt = map(x-> norm(R[x,:])>1e-12,1:size(R,1))
        U = U[:,flt]
        R = R[flt,:]

        if size(R,1)==0 || size(R,2) == 0
            return k=> (U,R)
        end

        (L_s,V_s) = lq(R)
        L = Matrix(L_s);
        V = Matrix(V_s)

        flt = map(x-> norm(L[:,x])>1e-12,1:size(L,2))
        V = V[flt,:]
        L = L[:,flt]
        
        U = U*L;

        k => (U,V)
    end

    kvs = Dict(pairs)
    totblock_inds = reduce(vcat,[[(k,i) for i in 1:size(kvs[k][1],2)] for k in keys(kvs)])

    #mapper = Map() do (k,i)

    blocks = map(totblock_inds) do (k,i)
        (U,V) = kvs[k]
        cur_left_blocks = blocked_left_blocks[left_group[k]]
        cur_right_blocks = blocked_right_blocks[right_group[k]]

        l = empty(cur_left_blocks[1][1])
        for j in 1:size(U,1)
            scal = U[j,i]
            abs(scal) < 1e-12 && continue

            sparsified_left = cur_left_blocks[j][1]
            for (k,b) in sparsified_left
                if k in keys(l)
                    axpy!(scal,b,l[k])
                else
                    l[k] = b*scal
                end
            end
        end

        r = empty(cur_right_blocks[1][4])
        for j in 1:size(V,2)
            scal = V[i,j]
            abs(scal) < 1e-12 && continue

            sparsified_right = cur_right_blocks[j][4]
            for (k,b) in sparsified_right
                if k in keys(r)
                    axpy!(scal,b,r[k])
                else
                    r[k] = b*scal
                end
            end
        end


        table = Tuple{Tuple{Int,Int},Tuple{Int,Int},Int,Tuple{Int,Int},Tuple{Int,Int},Int,Matrix{eltype(O)},Matrix{eltype(O)}}[]
        for ((q1,q2,q3,q4,q5),block_l) in l

            for ((q6,_q4,_q3,_q2,q7),block_r) in r
                _q4 == q4 && _q3 == q3 && _q2 == q2 || continue
                
                (d1_1,d2_1),br = ac2_blockstructure.blockstructure[q2]
                sl1_1 = left_ac2_untrp[q2][q1]
                sl1_2 = right_ac2_untrp[q2][q7]
                offset_1 = (sl1_1.start-1)+(sl1_2.start-1)*d1_1+(br.start-1)

                (d1_2,d2_2),br = ac2_blockstructure.blockstructure[q4]
                sl2_1 = left_ac2_untrp[q4][q5]
                sl2_2 = right_ac2_untrp[q4][q6]
                offset_2 = (sl2_1.start-1)+(sl2_2.start-1)*d1_2+(br.start-1)

                push!(table,((length(sl1_1),length(sl1_2)),(1,d1_1),offset_1,(length(sl2_1),length(sl2_2)),(1,d1_2),offset_2,block_l,block_r))
				

            end
        end

        #fast_tmp_1 = fast_init(codomain(l),codomain(r),storagetype(l))
        #fast_submult = LeftSubMult(space(l),ac2_structure)
        
        # transpose + temps
        #(l,r,(fast_tmp_1,fast_submult))
        table
    end

    #blocks = tcollect(mapper,totblock_inds)
    reduced_blocks = reduce(vcat,blocks)

    buffersize = maximum(map(reduced_blocks) do (size_1,stride_1,offset_1,size_2,stride_2,offset_2,left,right)
        max(size(left,1)*size_2[2],size_2[1]*size(right,2))
    end)
    fused_∂∂AC2(reduced_blocks,buffersize)
end

#=
BenchmarkTools.Trial: 43 samples with 1 evaluation.
 Range (min … max):  1.344 s …   1.692 s  ┊ GC (min … max): 3.40% … 3.47%
 Time  (median):     1.400 s              ┊ GC (median):    3.50%
 Time  (mean ± σ):   1.426 s ± 78.635 ms  ┊ GC (mean ± σ):  3.42% ± 0.25%

  ▃▃      █ ▁▁   ▁           ▁
  ██▁▇▁▁▇▇█▇██▄▁▄█▁▄▁▁▄▁▁▁▁▁▄█▁▁▁▁▁▁▄▁▁▁▁▁▁▁▁▄▁▁▄▁▁▁▁▁▁▁▁▁▄ ▁
  1.34 s         Histogram: frequency by time        1.69 s <

 Memory estimate: 1.90 GiB, allocs estimate: 607160.
 =#
function _reduce_ac2(table,x,basesize,buffersize)
    if length(table) <= basesize
        toret = zero(x)

        cur_buffer = storagetype(x)(undef,buffersize)

        for (size_1,stride_1,offset_1,size_2,stride_2,offset_2,left,right) in table
            v1 = StridedView(toret.data,size_1,stride_1,offset_1)
            v2 = StridedView(x.data,size_2,stride_2,offset_2)
            dst = StridedView(cur_buffer,(size(left,1),size(v2,2)),(1,size(left,1)))

            mul!(dst,StridedView(left),v2)
            mul!(v1,dst,StridedView(right),true,true)
        end

        return toret
    else

        spl = Int(ceil(length(table)/2));
        t = @Threads.spawn _reduce_ac2(table[1:spl],x,basesize,buffersize)
        toret = _reduce_ac2(view(table,spl+1:length(table)),x,basesize,buffersize)
        fast_axpy!(true,fetch(t),toret)
        return toret
    end
end

function (h::fused_∂∂AC2)(x)
    _reduce_ac2(h.table,x,ceil(length(h.table)/nthreads()),h.buffersize)
end
