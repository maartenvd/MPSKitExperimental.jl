struct FusedSparseBlock{E,O,Sp}
    domspaces::Vector{Sp}
    imspaces::Vector{Sp}

    pspace::Sp

    # Lmask, Ltensors, Opp, Rtensors, Rmask
    blocks :: Vector{Tuple{Vector{Bool},Vector{E},Union{E,O},Vector{E},Vector{Bool}}}

    function FusedSparseBlock{E,O,Sp}(domspaces::Vector{Sp},imspaces::Vector{Sp},pspace::Sp,
        blocks::Vector{Tuple{Vector{Bool},Vector{E},Union{E,O},Vector{E},Vector{Bool}}}) where {E,O,Sp}
        
        for o in blocks
            @assert sum(o[1])>0 && sum(o[end]) > 0
            @assert sum(o[1]) == length(o[2])
            @assert sum(o[end]) == length(o[end-1])

            if o[3] isa O
                @assert all(map(x->x==space(o[3],1),domspaces[o[1]]))
                @assert all(map(x->x==space(o[3],4),imspaces[o[end]]))
            end
        end

        new{E,O,Sp}(domspaces,imspaces,pspace,blocks);
    end
end

function Base.getproperty(x::FusedSparseBlock,s::Symbol)
    if s == :odim
        length(x.domspaces)
    else
        getfield(x,s)
    end
end


struct FusedMPOHamiltonian{E,O,Sp}
    data::Vector{FusedSparseBlock{E,O,Sp}}
end


function Base.getproperty(x::FusedMPOHamiltonian,s::Symbol)
    if s == :odim
        @assert false # after compression, the odim will be site dependent
        getproperty(x.data[1],s)
    else
        getfield(x,s)
    end
end

Base.length(h::FusedMPOHamiltonian) = length(h.data);
Base.getindex(x::FusedMPOHamiltonian,args...) = x.data[args...];