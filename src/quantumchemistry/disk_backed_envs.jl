
#=
only left/rightenvs are stored on disk (the entire finitemps is still kept in memory)
=#

struct DiskBackedEnvs{B<:FusedMPOHamiltonian,C} <: Cache
    opp::B #the operator

    ldependencies::Vector{C} #the data we used to calculate leftenvs/rightenvs
    rdependencies::Vector{C}

    leftenvs::Vector{Vector{C}}
    rightenvs::Vector{Vector{C}}

    memory_manager::DiskManager
end

Base.copy(d::DiskBackedEnvs) = @assert false;
Base.deepcopy(d::DiskBackedEnvs) = @assert false;
#Base.Filesystem.mktemp

#=
    we can surpisingly enough hook into the standard finite env!
=#
function disk_environments(state::FiniteMPS{S},ham::FusedMPOHamiltonian,dm::DiskManager) where S
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

    return disk_environments(state,ham,leftstart,rightstart,dm)
end


function disk_environments(state,opp,leftstart,rightstart,manager::DiskManager)
    leftenvs = Vector{typeof(leftstart)}(undef,length(state)+1);
    rightenvs = Vector{typeof(rightstart)}(undef,length(state)+1);

    vtype = eltype(leftstart);
    
    leftenvs[1] = [copy2disk(manager,l) for l in leftstart];
    rightenvs[end] = [copy2disk(manager,l) for l in rightstart];

    example = leftstart[1];
    for i in 1:length(state)
        leftenvs[i+1] = [copy2disk(manager,example) for j in opp[i].imspaces]
        rightenvs[i] = [copy2disk(manager,example) for j in opp[i].domspaces]
    end
    
    t = similar(state.AL[1]);

    return DiskBackedEnvs(opp,fill(t,length(state)),fill(t,length(state)),leftenvs,rightenvs,manager);
end

#notify the cache that we updated in-place, so it should invalidate the dependencies
function MPSKit.poison!(ca::DiskBackedEnvs,ind)
    ca.ldependencies[ind] = similar(ca.ldependencies[ind])
    ca.rdependencies[ind] = similar(ca.rdependencies[ind])
end

#rightenv[ind] will be contracteable with the tensor on site [ind]
function MPSKit.rightenv(ca::DiskBackedEnvs{O,E},ind,state)::Vector{E} where{O,E}
    a = findfirst(i -> !(state.AR[i] === ca.rdependencies[i]), length(state):-1:(ind+1))

    if !isnothing(a)
        a = length(state)-a+1

        #we need to recalculate
        for j = a:-1:ind+1
            out = TransferMatrix(state.AR[j],ca.opp[j],state.AR[j])*ca.rightenvs[j+1]
            for i in 1:length(out) 
                deallocate!(ca.memory_manager,ca.rightenvs[j][i]);
                ca.rightenvs[j][i] = copy2disk(ca.memory_manager,out[i])
            end
            ca.rdependencies[j] = state.AR[j]
        end
    end

    return ca.rightenvs[ind+1]
end

function MPSKit.leftenv(ca::DiskBackedEnvs{O,E},ind,state)::Vector{E} where{O,E}
    a = findfirst(i -> !(state.AL[i] === ca.ldependencies[i]), 1:(ind-1))

    if !isnothing(a)
        #we need to recalculate
        for j = a:ind-1
            out = ca.leftenvs[j]*TransferMatrix(state.AL[j],ca.opp[j],state.AL[j])
            for i in 1:length(out)
                deallocate!(ca.memory_manager,ca.leftenvs[j+1][i]);
                ca.leftenvs[j+1][i] = copy2disk(ca.memory_manager,out[i])
            end
            ca.ldependencies[j] = state.AL[j]
        end
    end

    return ca.leftenvs[ind]
end
