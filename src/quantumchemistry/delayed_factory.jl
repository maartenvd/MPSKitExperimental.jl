using TensorKit:FusionTreeDict, SectorDict, HomSpace,fusiontreetype

#fast_similar(t::TensorMap{S, N₁, N₂, I, SD, F₁, F₂}) where {S, N₁, N₂, I, SD, F₁, F₂} =
#    TensorMap{S, N₁, N₂, I, SD, F₁, F₂}(SD(t.data.keys,similar.(t.data.values)),t.codom,t.dom,t.rowr,t.colr)
#fast_copy(t::TensorMap{S, N₁, N₂, I, SD, F₁, F₂}) where {S, N₁, N₂, I, SD, F₁, F₂} =
#    TensorMap{S, N₁, N₂, I, SD, F₁, F₂}(SD(t.data.keys,copy.(t.data.values)),t.codom,t.dom,t.rowr,t.colr)


    


# creates an uninitialized tensormap at a later time
struct DelayedFact{T,S,N₁,N₂,A,B}
    homsp::TensorMapSpace{S,N₁,N₂}

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

