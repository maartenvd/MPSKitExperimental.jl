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

# The stacked AC2 halves are filled per fusion tree, which needs the row and column range of every fusion tree in its block
function blockstructure_dict(W::TensorKit.HomSpace)
    ss = TensorKit.sectorstructure(W)
    ds = TensorKit.degeneracystructure(W)
    return Dict(zip(collect(ss.blocksectors),ds.blockstructure))
end

# Rebuilds features that used to explicitly exist in TensorKit
# Per global charge, give a mapping from fusiontrees to the indices in the matrix.
function rowr_colr_from_fusionblockstructure(W::TensorKit.HomSpace)
    # untested with fusion multiplicities
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

#=
    The two-site operator in MPSKit's precomputed form (MPSKit.PrecomputedDerivative), applied by MPSKit's own code:
    the environments are contracted with the operators once per eigensolve, the middle link becomes one leg, and
    the action on AC2 is two block matrix products.

    The middle index runs over the bond states of the environment basis, C = X·Y, the bond MPSKit contracts over for
    the same hamiltonian. The difference is in building the two halves: here they are sparse scalar sums of
    per-channel env·operator tensors,

        (GL·O)_β = Σ_k X[k,β] (env_k·O_k),   (O·GR)_β = Σ_k Y[β,k] (O_k·env_k),

    instead of contractions with operator-valued MPO entries (where the dense block of V at the NC/CN switch is
    expensive for an ordinary MPO).

    Rank reducing C cannot make the middle leg shorter: it can only mix channels whose operators have the same middle
    space, and per such space X·Y already has the rank of C (checked for N2 cc-pVDZ, every bond).
=#

# the direct sum of `spaces`, with the range every summand occupies per sector; dual spaces give the dual sum
function _stacked(spaces::Vector{Sp}) where Sp
    if all(isdual,spaces)
        (M,slots) = _stacked(dual.(spaces))
        return (M',[Dict(dual(s) => r for (s,r) in sl) for sl in slots])
    end
    S = sectortype(Sp)
    total = Dict{S,Int}()
    slots = map(spaces) do V
        Dict(s => (o = get(total,s,0); total[s] = o + dim(V,s); (o+1):(o+dim(V,s))) for s in sectors(V))
    end
    M = Vect[S]((s => d for (s,d) in total)...)
    (M,slots)
end

# channels with the same operator (same value and spaces), as lists of positions in `chs`
function _opgroups(chs)
    gid = Dict{Any,Int}(); groups = Vector{Vector{Int}}()
    for (p,c) in enumerate(chs)
        g = get!(gid,(space(c.op),c.op.data)) do
            push!(groups,Int[]); length(groups)
        end
        push!(groups[g],p)
    end
    groups
end

# per operator group and middle bond state: the weighted channels, (positions, weights)
function _bystate(chs,groups,side)
    map(groups) do g
        d = Dict{Int,Tuple{Vector{Int},Vector{eltype(first(chs).lval)}}}()
        for p in g
            c = chs[p]
            idx,val = side === :right ? (c.ridx,c.rval) : (c.lidx,c.lval)
            for (β,w) in zip(idx,val)
                (ps,ws) = get!(() -> (Int[],eltype(val)[]),d,β)
                push!(ps,p); push!(ws,w)
            end
        end
        d
    end
end

# adds the blocks of a term t into the stacked tensor: every fusion tree goes to the range of its sector's slot
# along the stacked leg (rows if the stacked leg is in the codomain, columns if it is in the domain)
function _addslots!(bigblocks,t,plan,slot,dir)
    for (q,entries) in plan
        b = block(t,q); bb = bigblocks[q]
        for (r,base,sec) in entries
            sl = slot[sec]; n = length(r)
            dst = base + (first(sl)-1)*n
            if dir === :rows
                axpy!(true,view(b,r,:),view(bb,dst:(dst+n*length(sl)-1),:))
            else
                axpy!(true,view(b,:,r),view(bb,:,dst:(dst+n*length(sl)-1)))
            end
        end
    end
    bigblocks
end

# backend and allocator are the ones MPSKit's algorithms pass (DMRG2 hands in a BufferAllocator for the matvecs)
MPSKit.AC2_hamiltonian(pos::Int,below,ham::LinkMPOHamiltonian,above,cache;
                       backend = TensorOperations.DefaultBackend(),allocator = TensorOperations.DefaultAllocator(),kwargs...) =
    link_AC2_hamiltonian(pos,below,ham,cache;backend,allocator)
function link_AC2_hamiltonian(pos::Int,mps,ham::LinkMPOHamiltonian{E},cache;
                              backend = TensorOperations.DefaultBackend(),allocator = TensorOperations.DefaultAllocator()) where E
    le = leftenv(cache,pos,mps);
    re = rightenv(cache,pos+1,mps);
    T = scalartype(le[1])
    TT = tensormaptype(spacetype(le[1]),3,2,storagetype(le[1]))
    structure = Dict{Any,Any}()   # row/column ranges of every fusion tree in its block, per space
    ranges(W) = get!(() -> rowr_colr_from_fusionblockstructure(W),structure,W)
    nmid = length(ham.bondspaces[pos+1])
    # the stacked halves and the per-term pieces are temporaries (only the repartitioned halves are kept), so they
    # come from the allocator, as in MPSKit's _prepare_GL_O; the terms are therefore built one after the other
    cp = TensorOperations.allocator_checkpoint!(allocator)

    # GL·O, as MPSKit lays it out: (v_out ⊗ p_out ⊗ middle) ← (v_in ⊗ p_in). Per middle bond state β and per
    # operator O the left environments are first combined with the link scalars (small tensors), then
    # contracted with O once: Σ_k X[k,β] env_k·O_k = Σ_O (Σ_{k with O} X[k,β] env_k)·O.
    lchs = ham.channels[pos]
    ls = [_combine(le,c.lidx,c.lval) for c in lchs]
    lgroups = _opgroups(lchs)
    lterms = [(g,β,ps,ws) for (g,d) in zip(lgroups,_bystate(lchs,lgroups,:right)) for (β,(ps,ws)) in d]
    midspace = Vector{Any}(undef,nmid)
    for (g,β,_,_) in lterms; midspace[β] = space(lchs[first(g)].op,4); end
    (M,slots) = _stacked(identity.(midspace))
    l1 = first(ls); o1 = lchs[first(first(lterms)[1])].op
    GLO = TensorOperations.tensoralloc(TT,space(l1,1) ⊗ space(o1,2) ⊗ M ← domain(l1)[1] ⊗ domain(o1)[1],Val(true),allocator)
    zerovector!(GLO)
    (rowrL,_) = ranges(space(GLO))
    # the middle leg is the last codomain leg: a bond state is a contiguous range of rows of every fusion tree.
    # Where every fusion tree of a term goes only depends on the term's space, so that is worked out once per space.
    rowplan = Dict{Any,Any}()
    plan_rows(W) = get!(rowplan,W) do
        (rowr,_) = ranges(W)
        [q => [(rr,first(rowrL[q][f1]),f1.uncoupled[3]) for (f1,rr) in rowr[q]] for q in keys(rowr)]
    end
    GLOblocks = Dict(q => b for (q,b) in blocks(GLO))
    for (g,β,ps,ws) in lterms
        l = _combine(ls,ps,ws)
        op = lchs[first(g)].op
        glo = TensorOperations.tensoralloc(TT,space(l,1) ⊗ space(op,2) ⊗ space(op,4) ← domain(l)[1] ⊗ domain(op)[1],Val(true),allocator)
        @planar backend = backend allocator = allocator glo[-1 -2 -3; -4 -5] = l[-1 1; -4]*op[1 -2; -5 -3]
        _addslots!(GLOblocks,glo,plan_rows(space(glo)),slots[β],:rows)
        TensorOperations.tensorfree!(glo,allocator)
    end

    # O·GR: (v_in ⊗ p_in) ← (v_out ⊗ p_out ⊗ middle), the same way with the link scalars Y
    TR = tensormaptype(spacetype(re[1]),2,3,storagetype(re[1]))
    rchs = ham.channels[pos+1]
    rs = [_combine(re,c.ridx,c.rval) for c in rchs]
    rgroups = _opgroups(rchs)
    rterms = [(g,β,ps,ws) for (g,d) in zip(rgroups,_bystate(rchs,rgroups,:left)) for (β,(ps,ws)) in d]
    for (g,β,_,_) in rterms; midspace[β] = space(rchs[first(g)].op,1)'; end
    (Mr,rslots) = _stacked(identity.(midspace))
    r1 = first(rs); o2 = rchs[first(first(rterms)[1])].op
    OGR = TensorOperations.tensoralloc(TR,space(r1,1) ⊗ space(o2,3) ← domain(r1)[1] ⊗ space(o2,2)' ⊗ Mr,Val(true),allocator)
    zerovector!(OGR)
    (_,colrR) = ranges(space(OGR))
    # the middle leg is the last domain leg: a contiguous range of columns of every fusion tree
    colplan = Dict{Any,Any}()
    plan_cols(W) = get!(colplan,W) do
        (_,colr) = ranges(W)
        [q => [(cr,first(colrR[q][f2]),f2.uncoupled[3]) for (f2,cr) in colr[q]] for q in keys(colr)]
    end
    OGRblocks = Dict(q => b for (q,b) in blocks(OGR))
    for (g,β,ps,ws) in rterms
        r = _combine(rs,ps,ws)
        op = rchs[first(g)].op
        ogr = TensorOperations.tensoralloc(TR,space(r,1) ⊗ space(op,3) ← domain(r)[1] ⊗ space(op,2)' ⊗ space(op,1)',Val(true),allocator)
        @planar backend = backend allocator = allocator ogr[-1 -2; -4 -5 -3] = op[-3 -5; -2 1]*r[-1 1; -4]
        _addslots!(OGRblocks,ogr,plan_cols(space(ogr)),rslots[β],:cols)
        TensorOperations.tensorfree!(ogr,allocator)
    end

    Lp = repartition(MPSKit.fuse_legs(GLO,1,2),2,2;copy = true,backend,allocator)
    Rp = repartition(MPSKit.fuse_legs(OGR,2,1),2,2;copy = true,backend,allocator)
    TensorOperations.tensorfree!(GLO,allocator)
    TensorOperations.tensorfree!(OGR,allocator)
    TensorOperations.allocator_reset!(allocator,cp)
    return MPSKit.PrecomputedDerivative(Lp,Rp,backend,allocator)
end
