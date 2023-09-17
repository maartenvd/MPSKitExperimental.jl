using TensorKit:FusionTreeDict, SectorDict, HomSpace,fusiontreetype

fast_similar(t::TensorMap{S, N₁, N₂, I, SD, F₁, F₂}) where {S, N₁, N₂, I, SD, F₁, F₂} =
    TensorMap{S, N₁, N₂, I, SD, F₁, F₂}(SD(t.data.keys,similar.(t.data.values)),t.codom,t.dom,t.rowr,t.colr)
fast_copy(t::TensorMap{S, N₁, N₂, I, SD, F₁, F₂}) where {S, N₁, N₂, I, SD, F₁, F₂} =
    TensorMap{S, N₁, N₂, I, SD, F₁, F₂}(SD(t.data.keys,copy.(t.data.values)),t.codom,t.dom,t.rowr,t.colr)

function fast_axpy!(a,x,y)
    xk = x.data.keys;
    xv = x.data.values;
    yk = y.data.keys;
    yv = y.data.values;

    @assert length(xk) == length(yk) # otherwise can still define it efficiently - but more tricky
    for (vx,vy) in zip(xv,yv)
        axpy!(a,vx,vy)
    end

    y

end
# creates an uninitialized tensormap at a later time
struct DelayedFact{S,N₁,N₂,F₁,F₂,I,A,B}
    cod::ProductSpace{S,N₁}
    dom::ProductSpace{S,N₂}
    
    rowr::SectorDict{I, FusionTreeDict{F₁, UnitRange{Int}}}
    colr::SectorDict{I, FusionTreeDict{F₂, UnitRange{Int}}}
    
    rowdims::SectorDict{I, Int}
    coldims::SectorDict{I, Int}

    storage::A

    recyclers::ConcurrentStack{B}
end


function DelayedFactType(S,storage,N₁,N₂)
    I = sectortype(S);
    F₁ = fusiontreetype(I, N₁)
    F₂ = fusiontreetype(I, N₂)

    
    B = TensorMap{S, N₁, N₂, I, SectorDict{I,storage}, F₁, F₂}
    DelayedFact{S,N₁,N₂,F₁,F₂,I,typeof(storage),B}
end
function DelayedFact(homsp,storage)
    S = spacetype(homsp);

    (rowr_src,colr_src,rowdims_src,coldims_src) = calc_rowr_color(sectortype(homsp),codomain(homsp),domain(homsp));
    
    I = sectortype(S)
    N₁ = length(codomain(homsp));
    N₂ = length(domain(homsp));
    F₁ = fusiontreetype(I, N₁)
    F₂ = fusiontreetype(I, N₂)

    T = tensormaptype(S,N₁,N₂,storage);
    DelayedFact(codomain(homsp),domain(homsp),rowr_src,colr_src,rowdims_src,coldims_src,storage,ConcurrentStack{T}())
end

function free!(d::DelayedFact,t)
    push!(d.recyclers,t)
end

function (fact::DelayedFact{S,N₁,N₂,F₁,F₂,I,A,B})() where {S,N₁,N₂,F₁,F₂,I,A,B}
    m = maybepop!(fact.recyclers);
    if m isa Some
        return something(m)::B
    else
        # allocate the output tensormap
        keys = I[];
        vals = fact.storage[];
        for (c,rd) in fact.rowdims
            cd = fact.coldims[c];
            push!(keys,c);
            push!(vals,fact.storage(undef,rd,cd);)
        end
        data = SectorDict{I,fact.storage}(keys,vals);

        TensorMap{S, N₁, N₂, I, SectorDict{I,fact.storage}, F₁, F₂}(data, fact.cod, fact.dom, fact.rowr, fact.colr);
    end
end

function calc_rowr_color(I,codom,dom)
    # this _HAS_ to be the only way that rowr/colr in tensormaps are generated, otherwise everything breaks down
    N₁ = length(codom);
    N₂ = length(dom);

    F₁ = fusiontreetype(I, N₁)
    F₂ = fusiontreetype(I, N₂)
    
    rowr = SectorDict{I, FusionTreeDict{F₁, UnitRange{Int}}}()
    colr = SectorDict{I, FusionTreeDict{F₂, UnitRange{Int}}}()
    rowdims = SectorDict{I, Int}()
    coldims = SectorDict{I, Int}()
    if N₁ == 0 || N₂ == 0
        blocksectoriterator = (one(I),)
    elseif N₂ <= N₁
        blocksectoriterator = blocksectors(dom)
    else
        blocksectoriterator = blocksectors(codom)
    end
    for s1 in sectors(codom)
        for c in blocksectoriterator
            offset1 = get!(rowdims, c, 0)
            rowrc = get!(rowr, c) do
                FusionTreeDict{F₁, UnitRange{Int}}()
            end
            for f1 in fusiontrees(s1, c, map(isdual, codom.spaces))
                r = (offset1 + 1):(offset1 + dim(codom, s1))
                push!(rowrc, f1 => r)
                offset1 = last(r)
            end
            rowdims[c] = offset1
        end
    end
    for s2 in sectors(dom)
        for c in blocksectoriterator
            offset2 = get!(coldims, c, 0)
            colrc = get!(colr, c) do
                FusionTreeDict{F₂, UnitRange{Int}}()
            end
            for f2 in fusiontrees(s2, c, map(isdual, dom.spaces))
                r = (offset2 + 1):(offset2 + dim(dom, s2))
                push!(colrc, f2 => r)
                offset2 = last(r)
            end
            coldims[c] = offset2
        end
    end
    for c in blocksectoriterator
        dim1 = get!(rowdims, c, 0)
        dim2 = get!(coldims, c, 0)
        if dim1 == 0 || dim2 == 0
            delete!(rowr, c)
            delete!(colr, c)
            delete!(rowdims,c)
            delete!(coldims,c)
            #data[c] = f((dim1, dim2))
        end

    end
    
    # return TensorMap{S, N₁, N₂, I, SectorDict{I,A}, F₁, F₂}(data, codom, dom, rowr, colr)

    return (rowr,colr,rowdims,coldims)
end
