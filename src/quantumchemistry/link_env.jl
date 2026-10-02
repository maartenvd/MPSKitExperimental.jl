#=
    environments for a LinkMPOHamiltonian always live on disk
=#
MPSKit.environments(state::FiniteMPS,ham::LinkMPOHamiltonian) = disk_environments(state,ham)
function MPSKit.environments(below::FiniteMPS,ham::LinkMPOHamiltonian,above::FiniteMPS)
    below === above || throw(ArgumentError("LinkMPOHamiltonian environments require below === above"))
    return disk_environments(below,ham)
end

#=
    Growing an environment through site n: combine the stored environments per channel (axpys with Y or X),
    one contraction per channel with the mps and the channel operator, then scatter onto the bond states of
    the next link (axpys).
=#
function _combine(v,idx,val)
    # idx labels the indices you need to grab from v, val labels the values you need to multiply them with
    l = rmul!(copy(v[idx[1]]),val[1])
    for i in 2:length(idx)
        l = axpy!(val[i],v[idx[i]],l)
    end
    l
end

# we applied L, we applied O, we have yet to apply R
function left_channel_envs(v::Vector,h::LinkMPOHamiltonian,n::Int,A,Ab=A)
    Ab_flipped = convert(TensorMap,transpose(Ab',((1,3),(2,))))
    mapper = Map() do c
        l = _combine(v,c.lidx,c.lval)
        @planar allocator = malloc() y[-1 -2;-3] := l[4 2;1]*A[1 3;-3]*c.op[2 5;3 -2]*Ab_flipped[-1 5;4]
        y
    end
    tcollect(mapper,h.channels[n])
end

# same as left_channel_envs, but comming from the right
function right_channel_envs(v::Vector,h::LinkMPOHamiltonian,n::Int,A,Ab=A)
    Ab_flipped = convert(TensorMap,transpose(Ab',((1,3),(2,))))
    mapper = Map() do c
        r = _combine(v,c.ridx,c.rval)
        @planar allocator = malloc() nr[-1 -2;-3] := A[-1 2;1]*r[1 3;4]*c.op[-2 5;2 3]*Ab_flipped[4 5;-3]
        nr
    end
    tcollect(mapper,h.channels[n])
end

# we need to apply R on top of the left_channel_envs
# lists[i] gathers "which channels write to channel_i"
# if none write, it needs to be explicitly initialized to a zero
# another micro optimization is possible here - if the bond dimension was unchanged, we could re-use those tensors. But they probably live on disk anyway...
function _scatter(ys,lists,zerofor)
    out = Vector{eltype(ys)}(undef,length(lists))
    @floop for a in eachindex(lists)
        if isempty(lists[a])
            out[a] = zerofor(a)
        else
            (c,w) = lists[a][1]
            t = rmul!(copy(ys[c]),w)
            for (c,w) in Iterators.drop(lists[a],1)
                t = axpy!(w,ys[c],t)
            end
            out[a] = t
        end
    end
    out
end

# gather all channels that write to i - need this in _scatter
function _lists(chs::Vector{LinkChannel{E,O}},nstates,side) where {E,O}
    lists = [Tuple{Int,E}[] for _ in 1:nstates]
    for (p,c) in enumerate(chs)
        idx,val = side === :right ? (c.ridx,c.rval) : (c.lidx,c.lval)
        for (a,w) in zip(idx,val)
            push!(lists[a],(p,w))
        end
    end
    lists
end

function transfer_left(v::Vector,h::LinkMPOHamiltonian,n::Int,A,Ab=A)
    ys = left_channel_envs(v,h,n,A,Ab)
    states = h.bondspaces[n+1]
    _scatter(ys,_lists(h.channels[n],length(states),:right),
             a -> zeros(scalartype(A),space(Ab,3)'*states[a]'←space(A,3)'))
end

function transfer_right(v::Vector,h::LinkMPOHamiltonian,n::Int,A,Ab=A)
    rs = right_channel_envs(v,h,n,A,Ab)
    states = h.bondspaces[n]
    _scatter(rs,_lists(h.channels[n],length(states),:left),
             a -> zeros(scalartype(A),space(A,1)*states[a]←space(Ab,1)))
end
