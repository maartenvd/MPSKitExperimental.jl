using Serialization
#=
only left/rightenvs are stored on disk (the entire finitemps is still kept in memory)

Same caching logic as MPSKit's FiniteEnvironments (an environment is recomputed when a tensor it depends on is
no longer the one it was computed from), with MPSKit's own transfers; only the storage differs.
=#

mutable struct ManualDiskBackedEnvs{B,C,L,R} <: MPSKit.AbstractMPSEnvironments
    operator::B #the operator

    ldependencies::Vector{C} #the data we used to calculate leftenvs/rightenvs
    rdependencies::Vector{C}


    leftenvs::Vector{String} # list of files
    rightenvs::Vector{String} # list of files

    left_loaded::Tuple{Int,L}
    right_loaded::Tuple{Int,R}
end

Base.copy(d::ManualDiskBackedEnvs) = @assert false;
Base.deepcopy(d::ManualDiskBackedEnvs) = @assert false;
#Base.Filesystem.mktemp

# the boundary environments are MPSKit's own
function disk_environments(state::FiniteMPS,ham::FiniteMPOHamiltonian)
    envs = environments(state,ham,state)
    return disk_environments(state,ham,envs.GLs[1],envs.GRs[end])
end


function disk_environments(state,opp,leftstart,rightstart)

    leftenvs = [tempname() for i in 1:length(state)+1]
    rightenvs = [tempname() for i in 1:length(state)+1]

    serialize(leftenvs[1],leftstart)
    serialize(rightenvs[end],rightstart)

    t = similar(state.AL[1]);

    envs = ManualDiskBackedEnvs(opp,fill(t,length(state)),fill(t,length(state)),leftenvs,rightenvs,(1,leftstart),(length(state)+1,rightstart));
    # tempname only cleans up at exit, which lets long runs (tempdir is often in ram) fill up the tempdir
    return finalizer(envs) do e
        foreach(f -> rm(f; force = true), e.leftenvs)
        foreach(f -> rm(f; force = true), e.rightenvs)
    end
end

#notify the cache that we updated in-place, so it should invalidate the dependencies
function MPSKit.poison!(ca::ManualDiskBackedEnvs,ind)
    ca.ldependencies[ind] = similar(ca.ldependencies[ind])
    ca.rdependencies[ind] = similar(ca.rdependencies[ind])
end

function load_left!(m::ManualDiskBackedEnvs,ind)
    if m.left_loaded[1] != ind
        m.left_loaded = (ind,deserialize(m.leftenvs[ind]))
    end
    return m.left_loaded[2]
end

function load_right!(m::ManualDiskBackedEnvs,ind)
    if m.right_loaded[1] != ind
        m.right_loaded = (ind,deserialize(m.rightenvs[ind]))
    end
    return m.right_loaded[2]
end

function store_right!(m::ManualDiskBackedEnvs,v,ind)
    serialize(m.rightenvs[ind],v)
    m.right_loaded = (ind,v)
end

function store_left!(m::ManualDiskBackedEnvs,v,ind)
    serialize(m.leftenvs[ind],v)
    m.left_loaded = (ind,v)
end

# newer MPSKit versions pass backend/allocator; the transfers here use the defaults, which every version accepts
#rightenv[ind] will be contracteable with the tensor on site [ind]
function MPSKit.rightenv(ca::ManualDiskBackedEnvs,ind,state;kwargs...)
    a = findfirst(i -> !(state.AR[i] === ca.rdependencies[i]), length(state):-1:(ind+1))

    if !isnothing(a)
        a = length(state)-a+1

        #we need to recalculate
        for j = a:-1:ind+1
            store_right!(ca,MPSKit.TransferMatrix(state.AR[j],ca.operator[j],state.AR[j])*load_right!(ca,j+1),j)
            ca.rdependencies[j] = state.AR[j]
        end
    end

    return load_right!(ca,ind+1)
end

function MPSKit.leftenv(ca::ManualDiskBackedEnvs,ind,state;kwargs...)
    a = findfirst(i -> !(state.AL[i] === ca.ldependencies[i]), 1:(ind-1))

    if !isnothing(a)
        #we need to recalculate
        for j = a:ind-1
            store_left!(ca,load_left!(ca,j)*MPSKit.TransferMatrix(state.AL[j],ca.operator[j],state.AL[j]),j+1)
            ca.ldependencies[j] = state.AL[j]
        end
    end

    return load_left!(ca,ind)
end
