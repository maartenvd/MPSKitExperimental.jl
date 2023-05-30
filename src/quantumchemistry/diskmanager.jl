using Mmap;

# not thread safe at all
struct DiskManager
    globdat::Vector{UInt8} # memory mapped
    blobs::Vector{Tuple{UInt64,UInt64}} # concurrent blocks of memory
end



Base.copy(d::DiskManager) = @assert false;
Base.deepcopy(d::DiskManager) = @assert false;

DiskManager() = DiskManager(UInt8[],Tuple{Int,Int}[(1,0)]);
DiskManager(filename::String,size) = DiskManager(Mmap.open(io->mmap(io,Vector{UInt8},(size,)),filename,"w+"),[(1,size)])

function copy2disk(dm::DiskManager,t::TensorMap{S,N1,N2,C,T,F1,F2}) where {S,N1,N2,C,T,F1,F2}
    E = eltype(t);
    if t.data isa DenseMatrix
        ndat = allocate!(dm,E,size(t.data)...);
        copy!(ndat,t.data);
        TensorMap{S,N1,N2,C,T,F1,F2}(ndat,codomain(t),domain(t))
    else
        I = sectortype(t);
        ndat = TensorKit.SectorDict{I,Matrix{E}}();

        for (k,v) in t.data
            nb = allocate!(dm,E,size(v)...);
            copy!(nb,v);
            ndat[k] = nb;
        end

        TensorMap{S,N1,N2,C,T,F1,F2}(ndat,codomain(t),domain(t),t.rowr,t.colr)
    end
end
function deallocate!(dm::DiskManager)
    empty!(dm.blobs)
    push!(dm.blobs,(1,length(dm.globdat)))
    dm
end
function deallocate!(dm::DiskManager,data::AbstractTensorMap)
    for (k,v) in blocks(data)
        deallocate!(dm,v)
    end
end
function deallocate!(dm::DiskManager,data::Array{T,N}) where{T,N}
    cur_size = sizeof(T)*prod(size(data));
    cur_offset = pointer(data)-pointer(dm.globdat,1)+1;
    nhit = searchsortedfirst(dm.blobs,(cur_offset,cur_size),by= x->x[1]);

    leftmelt = nhit > 1 && sum(dm.blobs[nhit-1]) == cur_offset
    rightmelt = nhit <= length(dm.blobs) && dm.blobs[nhit][1] == cur_offset+cur_size

    if !leftmelt && !rightmelt
        insert!(dm.blobs,nhit,(cur_offset,cur_size))
    elseif !leftmelt
        dm.blobs[nhit] = (cur_offset,cur_size + dm.blobs[nhit][2]);
    elseif !rightmelt
        (a,b) = dm.blobs[nhit-1]
        dm.blobs[nhit-1] = (a,b+cur_size)
    else
        (a,b) = dm.blobs[nhit-1]
        dm.blobs[nhit-1] = (a,b+cur_size+dm.blobs[nhit][2]) ;
        deleteat!(dm.blobs,nhit)
    end
end

function allocate!(dm::DiskManager,T,dims::Vararg{Int,N})::Array{T,N} where N
    totalsize = sizeof(T)*prod(dims);
    
    hit = findfirst(x->x[2]>=totalsize,dm.blobs)
    isnothing(hit) && throw(ErrorException("out of memory"))
    
    (offset,blobsize) = dm.blobs[hit];

    new_blobsize = blobsize-totalsize;
    new_offset = offset+totalsize;

    if new_blobsize == 0
        deleteat!(dm.blobs,hit)
    else
        dm.blobs[hit] = (new_offset,new_blobsize)
    end

    unsafe_wrap(Array{T,N},convert(Ptr{T}, pointer(dm.globdat,offset)),dims)
end