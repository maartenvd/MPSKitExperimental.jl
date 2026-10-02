struct link_∂∂AC{A}
    blocks::A
end

MPSKit.AC_hamiltonian(pos::Int,below,ham::LinkMPOHamiltonian,above,cache;kwargs...) = link_AC_hamiltonian(pos,below,ham,cache)
function link_AC_hamiltonian(pos::Int,mps,ham::LinkMPOHamiltonian,cache)
    le = leftenv(cache,pos,mps);
    re = rightenv(cache,pos,mps);

    # we have the environments, we now need to combine per channel (apply L and R)
    process = Map() do c
        (_combine(le,c.lidx,c.lval),c.op,_combine(re,c.ridx,c.rval))
    end
    blocks = tcollect(process,ham.channels[pos])

    # drop empty blocks - though they really shouldn't exist anyway
    filter!(blocks) do (l,e,r)
        !(norm(l)<1e-12 || norm(r)<1e-12)
    end
    
    link_∂∂AC(blocks)
end

function (h::link_∂∂AC)(x)
    @floop for (l,e,r) in h.blocks
        @init t = similar(x)
        
        @planar allocator=malloc() t[-1 -2;-3] = l[-1 5; 4] * x[4 2; 1] * e[5 -2; 2 3] * r[1 3; -3]

        @reduce() do (toret = zero(x); t)
            axpy!(true,t,toret);
            toret
        end
    end
    
    toret
end

Base.:*(a::link_∂∂AC,v) = a(v)

# ugly - inconsistent with MPOHamiltonian
MPSKit.expectation_value(st::FiniteMPS,th::LinkMPOHamiltonian,envs = environments(st,th)) =
    dot(st.AC[1],link_AC_hamiltonian(1,st,th,envs)(st.AC[1]))/dot(st.AC[1],st.AC[1])

# Precalculates how GLW, GRW acts on AC2, so not a good idea if this precontraction is suboptimal!
struct link_∂∂AC2{A,W}
    table::A
    buffersize::Int
    space::W # the table indexes raw data, so it is only valid for this space
end

# The two site effective hamiltonian goes a bit wild, in that I really dig "this piece of matrix needs to be combined with this piece of matrix and this piece, to then go there"
# that requires explicity decomposing into fusion trees
function blockstructure_dict(W::TensorKit.HomSpace)
    ss = TensorKit.sectorstructure(W)
    ds = TensorKit.degeneracystructure(W)
    return Dict(zip(collect(ss.blocksectors),ds.blockstructure))
end

# Rebuilds features that used to explicitly exist in TensorKit
# Per global charge, give a mapping from fusiontrees to the indices in the matrix.
function rowr_colr_from_fusionblockstructure(W::TensorKit.HomSpace)
    # the subblock labels (_untrip_row/_untrip_col) are sectors only, without the vertex labels of fusion multiplicities
    FusionStyle(sectortype(W)) isa GenericFusion &&
        throw(ArgumentError("sectors with fusion multiplicities are not supported"))
    ss = TensorKit.sectorstructure(W)
    ds = TensorKit.degeneracystructure(W)
    blockstructure = blockstructure_dict(W)
    trees = collect(ss.fusiontrees)
    F₁ = typeof(first(trees)[1])
    F₂ = typeof(first(trees)[2])
    S = sectortype(W)
    rowr = Dict{S,Dict{F₁,UnitRange{Int}}}()
    colr = Dict{S,Dict{F₂,UnitRange{Int}}}()
    N1 = numout(W)
    N2 = numin(W)
    for ((f1,f2),(sz,st,o)) in zip(trees,ds.subblockstructure)
        (block_sz,block_range) = blockstructure[f1.coupled]
        block_range_start = block_range[1]
        
        #(subsz, substr, totaloffset)

        # reshape back to codomain/domain:
        i = mod(o+1-block_range_start,block_sz[1])+1
        j = (o+1-block_range_start)÷block_sz[1]+1

        irange = i:i+prod(sz[1:N1])-1
        jrange = j:j+prod(sz[N1+1:N1+N2])-1

        if !(f1.coupled in keys(rowr))
            rowr[f1.coupled] = Dict{F₁,UnitRange{Int}}()
            colr[f1.coupled] = Dict{F₂,UnitRange{Int}}()
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

# Per coupled sector, merge the row (or column) ranges of all fusion trees with the same label into one range.
# The AC2 table treats that range as a single subblock, which is only right if those trees sit next to each
# other in the block, so that is checked.
# (It should be explicitly imposed by tensorkit design - and by now mpskit also relies on this trick)
function _merge_ranges(r::Dict{S},label,::Type{L}) where {S,L}
    
    out = Dict{S,Dict{L,UnitRange{Int}}}()

    for (c,trees) in r
        merged = get!(out,c,Dict{L,UnitRange{Int}}())
        
        total = Dict{L,Int}()
        
        for (f,range) in trees
            q = label(f)
            merged[q] = haskey(merged,q) ? (min(first(range),first(merged[q])):max(last(range),last(merged[q]))) : range
            total[q] = get(total,q,0) + length(range)
        end
        
        for (q,range) in merged
            length(range) == total[q] || error("fusion trees with label $q are not contiguous in the block of $c")
        end
    end
    return out
end

# label: the last uncoupled sector (the physical sector, for the AC2 codomain)
_untrip_row(rowr::Dict{S}) where S = _merge_ranges(rowr,f -> f.uncoupled[end],S)
# label: (last uncoupled, last inner line, second to last uncoupled)
_untrip_col(colr::Dict{S}) where S = _merge_ranges(colr,f -> (f.uncoupled[end],f.innerlines[end],f.uncoupled[end-1]),Tuple{S,S,S})


#---------------------------

# per channel, applies L, then O, then pulls that apart into actions on fusiontrees
function _leftblock(chs::Vector{<:LinkChannel},le)
    blocked_left_blocks = map(chs) do c
        l = _combine(le,c.lidx,c.lval)
        e = c.op

        @planar allocator=malloc cle[-1 -2;-3 -4 -5] := l[-1 1;-3]*e[1 -2;-4 -5]
        
        (rowr,colr) = rowr_colr_from_fusionblockstructure(space(cle))
        sparsified = Dict{Tuple{sectortype(l),sectortype(l),sectortype(l),sectortype(l),sectortype(l)},Matrix{eltype(l)}}()

        untr_col = _untrip_col(colr)
        untr_row = _untrip_row(rowr)

        for (q2,b) in blocks(cle)
            for (q1,rowrange) in untr_row[q2], ((q3,q4,q5),colrange) in untr_col[q2]
                norm(b[rowrange,colrange],Inf) < 1e-12 && continue
                
                sparsified[(q1,q2,q3,q4,q5)] = copy(b[rowrange,colrange])
            end
        end
        
        # c.k is the index into the channel
        (sparsified,c.k)
    end

    filter!(blocked_left_blocks) do (l,k)
        !isempty(l)
    end
    
    return blocked_left_blocks
end

# the right counterpart to _leftblock
function _rightblock(chs::Vector{<:LinkChannel},re)
    blocked_right_blocks = map(chs) do c
        r = _combine(re,c.ridx,c.rval)
        e = c.op

        @planar allocator=malloc cre[-1 -2 -3;-4 -5] := r[-1 1;-4]*e[-3 -5;-2 1]
        
        (rowr,colr) = rowr_colr_from_fusionblockstructure(space(cre))
        sparsified = Dict{Tuple{sectortype(r),sectortype(r),sectortype(r),sectortype(r),sectortype(r)},Matrix{eltype(r)}}()

        untr_col = _untrip_row(colr)
        untr_row = _untrip_col(rowr)

        for (q2,b) in blocks(cre)
            for (q1,colrange) in untr_col[q2], ((q3,q4,q6),rowrange) in untr_row[q2]
                norm(b[rowrange,colrange],Inf) < 1e-12 && continue
                
                sparsified[(q6,q4,q3,q2,q1)] = copy(b[rowrange,colrange])
            end
        end

        (c.k,sparsified)
    end

    filter!(blocked_right_blocks) do (k,r)
        !isempty(r)
    end
    
    return blocked_right_blocks
end

#=
    The two-site operator is Σ_{a,b} d[a,b] cle_a ⊗ cre_b over left blocks a and right blocks b, with d the link
    between the two sites.

    cle_a and cre_b are block sparse: dictionaries of sector subblocks, keyed by (q1,q2,q3,q4,q5) on the left and
    (q6,q4,q3,q2,q7) on the right. A left and a right subblock that agree on (q2,q3,q4) form one table entry, i.e.
    one pair of GEMMs. We can rank reveal to make the mpo cheaper to apply:

        Σ_{a,b} d[a,b] cle_a[KL] ⊗ cre_b[KR] = Σ_t (Σ_a u_t[a] cle_a[KL]) ⊗ (Σ_b v_t[b] cre_b[KR]),

    with d restricted to the blocks that have KL and the blocks that have KR. 
=#

# the rank revealing qr
function _pqr(d)
    F = qr(d,ColumnNorm())
    r = count(x->abs(x)>1e-12,diag(F.R))
    (Matrix(F.Q)[:,1:r],F.R[1:r,:][:,invperm(F.p)])
end

# Σ_i w[i] * sparsified[idx[i]][key], or nothing if every weight vanishes
function _combine_subblock(sparsified,idx,w,key)
    out = nothing
    for (i,j) in enumerate(idx)
        abs(w[i]) < 1e-12 && continue
        b = sparsified[j][key]
        out = isnothing(out) ? b*w[i] : axpy!(w[i],b,out)
    end
    out
end

MPSKit.AC2_hamiltonian(pos::Int,below,ham::LinkMPOHamiltonian,above,cache;kwargs...) = link_AC2_hamiltonian(pos,below,ham,cache)
function link_AC2_hamiltonian(pos::Int,mps,ham::LinkMPOHamiltonian{E,O,Sp},cache) where {E,O,Sp}

    le = leftenv(cache,pos,mps);
    re = rightenv(cache,pos+1,mps);

    p1 = ham.pspaces[pos];
    p2 = ham.pspaces[pos+1];

    v1 = left_virtualspace(mps,pos);
    v2 = right_virtualspace(mps,pos+1);

    ac2_structure = v1*p1 ← v2*(p2)'

    S = sectortype(ac2_structure)

    ac2_blockstructure = blockstructure_dict(ac2_structure)
    (rowr_ac2,colr_ac2) = rowr_colr_from_fusionblockstructure(ac2_structure)
    left_ac2_untrp = _untrip_row(rowr_ac2)
    right_ac2_untrp = _untrip_row(colr_ac2)
    
    
    blocked_left_blocks = _leftblock(ham.channels[pos],le)
    blocked_right_blocks = _rightblock(ham.channels[pos+1],re)
    
    lsparse = [b[1] for b in blocked_left_blocks]; lchannel = [b[2] for b in blocked_left_blocks]
    rsparse = [b[2] for b in blocked_right_blocks]; rchannel = [b[1] for b in blocked_right_blocks]

    # which blocks have a given subblock key
    K = NTuple{5,S}
    lrows = Dict{K,Vector{Int}}(); rcols = Dict{K,Vector{Int}}()
    for (a,sp) in enumerate(lsparse), key in keys(sp); push!(get!(lrows,key,Int[]),a); end
    for (b,sp) in enumerate(rsparse), key in keys(sp); push!(get!(rcols,key,Int[]),b); end

    # right keys (q6,q4,q3,q2,q7) by the (q2,q3,q4) a left key (q1,q2,q3,q4,q5) has to match
    rbymatch = Dict{NTuple{3,S},Vector{K}}()
    for key in keys(rcols); push!(get!(rbymatch,(key[4],key[3],key[2]),K[]),key); end

    # compatible key pairs, grouped by the blocks they see
    groups = Dict{Tuple{Vector{Int},Vector{Int}},Vector{Tuple{K,K}}}()
    for (kl,rows) in lrows, kr in get(rbymatch,(kl[2],kl[3],kl[4]),K[])
        push!(get!(groups,(rows,rcols[kr]),Tuple{K,K}[]),(kl,kr))
    end

    C = ham.links[pos+1]
    table = Tuple{Tuple{Int,Int},Tuple{Int,Int},Int,Tuple{Int,Int},Tuple{Int,Int},Int,Matrix{eltype(O)},Matrix{eltype(O)}}[]
    for ((rows,cols),pairs) in groups

        #d is the dense subblock connecting these two
        d = E[C[lchannel[a],rchannel[b]] for a in rows, b in cols]
        iszero(d) && continue
        (U,V) = _pqr(d)
        
        # apply U and V, to get the sparsest table
        for t in axes(U,2)
            lcomb = Dict(kl => _combine_subblock(lsparse,rows,view(U,:,t),kl) for kl in unique(first.(pairs)))
            rcomb = Dict(kr => _combine_subblock(rsparse,cols,view(V,t,:),kr) for kr in unique(last.(pairs)))
            for (kl,kr) in pairs
                block_l = lcomb[kl]; block_r = rcomb[kr]
                (isnothing(block_l) || isnothing(block_r)) && continue
                (q1,q2,q3,q4,q5) = kl
                (q6,_,_,_,q7) = kr

                (d1_1,d2_1),br = ac2_blockstructure[q2]
                sl1_1 = left_ac2_untrp[q2][q1]
                sl1_2 = right_ac2_untrp[q2][q7]
                offset_1 = (sl1_1.start-1)+(sl1_2.start-1)*d1_1+(br.start-1)

                (d1_2,d2_2),br = ac2_blockstructure[q4]
                sl2_1 = left_ac2_untrp[q4][q5]
                sl2_2 = right_ac2_untrp[q4][q6]
                offset_2 = (sl2_1.start-1)+(sl2_2.start-1)*d1_2+(br.start-1)

                push!(table,((length(sl1_1),length(sl1_2)),(1,d1_1),offset_1,(length(sl2_1),length(sl2_2)),(1,d1_2),offset_2,block_l,block_r))
            end
        end
    end

    buffersize = maximum(table;init=0) do (size_1,stride_1,offset_1,size_2,stride_2,offset_2,left,right)
        size(left,1)*size_2[2]
    end
    link_∂∂AC2(table,buffersize,ac2_structure)
end


function _reduce_ac2(table,x,basesize,buffersize)
    if length(table) <= basesize
        toret = zero(x)

        cur_buffer = tensoralloc(storagetype(x), buffersize, Val(true), malloc)

        for (size_1,stride_1,offset_1,size_2,stride_2,offset_2,left,right) in table
            v1 = StridedView(toret.data,size_1,stride_1,offset_1)
            v2 = StridedView(x.data,size_2,stride_2,offset_2)
            dst = StridedView(cur_buffer,(size(left,1),size(v2,2)),(1,size(left,1)))

            mul!(dst,StridedView(left),v2)
            mul!(v1,dst,StridedView(right),true,true)
        end

        tensorfree!(cur_buffer, malloc)

        return toret
    else

        spl = Int(ceil(length(table)/2));
        t = @Threads.spawn _reduce_ac2(view(table,1:spl),x,basesize,buffersize)
        toret = _reduce_ac2(view(table,spl+1:length(table)),x,basesize,buffersize)
        axpy!(true,fetch(t),toret)
        return toret
    end
end

function (h::link_∂∂AC2)(x)
    space(x) == h.space || throw(SpaceMismatch("AC2 space $(space(x)) does not match $(h.space)"))
    _reduce_ac2(h.table,x,ceil(length(h.table)/nthreads()),h.buffersize)
end

Base.:*(a::link_∂∂AC2,v) = a(v)
