using Serialization
#=
only left/rightenvs are stored on disk (the entire finitemps is still kept in memory)
=#

mutable struct ManualDiskBackedEnvs{B<:FusedMPOHamiltonian,C} <: Cache
    opp::B #the operator

    ldependencies::Vector{C} #the data we used to calculate leftenvs/rightenvs
    rdependencies::Vector{C}


    leftenvs::Vector{String} # list of files
    rightenvs::Vector{String} # list of files

    left_loaded::Tuple{Int,Vector{C}}
    right_loaded::Tuple{Int,Vector{C}}
end

Base.copy(d::ManualDiskBackedEnvs) = @assert false;
Base.deepcopy(d::ManualDiskBackedEnvs) = @assert false;
#Base.Filesystem.mktemp

#=
    we can surpisingly enough hook into the standard finite env!
=#
function disk_environments(state::FiniteMPS{S},ham::FusedMPOHamiltonian) where S
    lll = l_LL(state);rrr = r_RR(state)
    rightstart = Vector{S}();leftstart = Vector{S}()

    for (i,sp) in enumerate(ham[1].domspaces)
        util_left = Tensor(x->storagetype(S)(undef,x),sp'); fill_data!(util_left,one);
        @plansor ctl[-1 -2; -3]:= lll[-1;-3]*util_left[-2]
        
        if i != 1
            ctl = zero(ctl)
        end

        push!(leftstart,ctl)
    end

    for (i,sp) in enumerate(ham[length(state)].imspaces)
        util_right = Tensor(x->storagetype(S)(undef,x),sp'); fill_data!(util_right,one);
        @plansor ctr[-1 -2; -3]:= rrr[-1;-3]*util_right[-2]

        if i != length(ham[length(state)].imspaces)
            ctr = zero(ctr)
        end

        push!(rightstart,ctr)
    end

    return disk_environments(state,ham,leftstart,rightstart)
end


function disk_environments(state,opp,leftstart,rightstart)

    leftenvs = [mktemp()[1] for i in 1:length(state)+1]
    rightenvs = [mktemp()[1] for i in 1:length(state)+1]
    
    serialize(leftenvs[1],leftstart)
    serialize(rightenvs[end],rightstart)
    
    t = similar(state.AL[1]);

    return ManualDiskBackedEnvs(opp,fill(t,length(state)),fill(t,length(state)),leftenvs,rightenvs,(1,leftstart),(length(state)+1,rightstart));
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

#rightenv[ind] will be contracteable with the tensor on site [ind]
function MPSKit.rightenv(ca::ManualDiskBackedEnvs{O,E},ind,state)::Vector{E} where{O,E}
    a = findfirst(i -> !(state.AR[i] === ca.rdependencies[i]), length(state):-1:(ind+1))

    if !isnothing(a)
        a = length(state)-a+1

        #we need to recalculate
        for j = a:-1:ind+1
            store_right!(ca,TransferMatrix(state.AR[j],ca.opp[j],state.AR[j])*load_right!(ca,j+1),j)
            ca.rdependencies[j] = state.AR[j]
        end
    end

    return load_right!(ca,ind+1)
end

function MPSKit.leftenv(ca::ManualDiskBackedEnvs{O,E},ind,state)::Vector{E} where{O,E}
    a = findfirst(i -> !(state.AL[i] === ca.ldependencies[i]), 1:(ind-1))

    if !isnothing(a)
        #we need to recalculate
        for j = a:ind-1
            store_left!(ca,load_left!(ca,j)*TransferMatrix(state.AL[j],ca.opp[j],state.AL[j]),j+1)
            ca.ldependencies[j] = state.AL[j]
        end
    end

    return load_left!(ca,ind)
end
