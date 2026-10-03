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

#=
    The order MPSKit uses for a sparse MPO: contract every incoming bond state with the mps tensor once, apply the
    operators, and contract every outgoing bond state with the conjugate tensor once. The first contraction leaves
    its result in the layout the operators need, so per channel there is only a combination of incoming bond
    states (axpys with the link scalars) and one matrix product with the operator; the outgoing bond states are
    again scalar combinations. The large contractions (with their permutations) happen once per incoming and once
    per outgoing bond state, independent of the number of channels, and operators only ever meet scalar
    combinations, never operator-valued MPO entries.
=#
function transfer_left(v::Vector,h::LinkMPOHamiltonian,n::Int,A,Ab=A)
    chs = h.channels[n]
    Ab_flipped = convert(TensorMap,transpose(Ab',((1,3),(2,))))
    states = h.bondspaces[n+1]

    # every incoming bond state times A, as (v_out, bra) ← (phys, chan)
    reads = sort!(unique!(reduce(vcat,[c.lidx for c in chs];init=Int[])))
    vAs = tcollect(Map(a -> (@planar t[-1 -2; -3 -4] := v[a][-2 -4; 1]*A[1 -3; -1]; t)),reads)
    vA = Dict(zip(reads,vAs))

    # per channel: (v_out, bra) ← (chan_out, p_out)
    zs = tcollect(Map(c -> _combine(vA,c.lidx,c.lval)*c.lop),chs)

    lists = _lists(chs,length(states),:right)
    out = Vector{eltype(v)}(undef,length(states))
    @floop for b in eachindex(lists)
        if isempty(lists[b])
            out[b] = zeros(scalartype(A),space(Ab,3)'*states[b]'←space(A,3)')
        else
            zb = _combine(zs,first.(lists[b]),last.(lists[b]))
            @planar y[-1 -2; -3] := zb[-3 1; -2 2]*Ab_flipped[-1 2; 1]
            out[b] = y
        end
    end
    out
end

function transfer_right(v::Vector,h::LinkMPOHamiltonian,n::Int,A,Ab=A)
    chs = h.channels[n]
    Ab_flipped = convert(TensorMap,transpose(Ab',((1,3),(2,))))
    states = h.bondspaces[n]

    # A times every incoming bond state, as (phys, chan) ← (v_out, bra)
    reads = sort!(unique!(reduce(vcat,[c.ridx for c in chs];init=Int[])))
    Avs = tcollect(Map(b -> (@planar t[-1 -2; -3 -4] := A[-3 -1; 1]*v[b][1 -2; -4]; t)),reads)
    Av = Dict(zip(reads,Avs))

    # per channel: (chan_left, p_out) ← (v_out, bra)
    zs = tcollect(Map(c -> c.op*_combine(Av,c.ridx,c.rval)),chs)

    lists = _lists(chs,length(states),:left)
    out = Vector{eltype(v)}(undef,length(states))
    @floop for a in eachindex(lists)
        if isempty(lists[a])
            out[a] = zeros(scalartype(A),space(A,1)*states[a]←space(Ab,1))
        else
            za = _combine(zs,first.(lists[a]),last.(lists[a]))
            @planar y[-1 -2; -3] := za[-2 1; -1 2]*Ab_flipped[2 1; -3]
            out[a] = y
        end
    end
    out
end
