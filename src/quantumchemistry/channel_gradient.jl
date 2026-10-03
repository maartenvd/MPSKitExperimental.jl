#=
    In quantum chemistry (and actually generically, noone is stopping you) we're able to put all degrees of freedom on the scalar weights of the channels.
    And so, we are able to evaluate the gradient of the energy with respect to all degrees of freedom in the hamiltonian automagically.
    And in turn, we're able to construct the derivative with respect to orbital rotations.

    How it works is quite magical (<3 claudy). 
        We construct a dummy type and then give that to the quantum chemistry hamiltonian builder.
        We construct these dummy environments
        We know the backwards derivative wrt to every parameters
        It really is just a manual backprop
    
    I wouldn't really bother trying to verify the code, as there is a very easy check. 
    If the quantum chemistry hamiltonian is correct, we can plug in random coefficients and compare the energy we get with what the RDM's predict.
=#

"""
    LinComb{T}

`c + Σ val[i] * θ[idx[i]]`, a scalar that is linear in some parameters θ. Products are only defined when one of
the factors is a constant, so a hamiltonian built from it is linear in θ by construction of every single entry.
"""
struct LinComb{T<:Real} <: Number
    c::T
    idx::Vector{Int}    # sorted, unique
    val::Vector{T}
end
LinComb{T}(x::Real) where T = LinComb{T}(T(x),Int[],T[])
LinComb{T}(x::LinComb) where T = LinComb{T}(T(x.c),x.idx,Vector{T}(x.val))
LinComb(x::LinComb) = x
lincomb_param(::Type{T},p::Int) where T = LinComb{T}(zero(T),[p],[one(T)])

Base.convert(::Type{LinComb{T}},x::Real) where T = LinComb{T}(x)
Base.convert(::Type{LinComb{T}},x::LinComb) where T = LinComb{T}(x)
Base.promote_rule(::Type{LinComb{T}},::Type{S}) where {T,S<:Real} = LinComb{promote_type(T,S)}
Base.promote_rule(::Type{LinComb{T}},::Type{LinComb{S}}) where {T,S} = LinComb{promote_type(T,S)}
Base.zero(::Type{LinComb{T}}) where T = LinComb{T}(zero(T))
Base.one(::Type{LinComb{T}}) where T = LinComb{T}(one(T))
Base.zero(x::LinComb) = zero(typeof(x))
Base.one(x::LinComb) = one(typeof(x))
isconstant(x::LinComb) = isempty(x.idx)
Base.iszero(x::LinComb) = iszero(x.c) && all(iszero,x.val)
# only used to decide what is structurally zero
Base.abs(x::LinComb) = abs(x.c) + sum(abs,x.val;init=zero(x.c))
Base.abs2(x::LinComb) = abs(x)^2
LinearAlgebra.norm(x::LinComb,p::Real=2) = abs(x)
Base.show(io::IO,x::LinComb) = print(io,x.c," + Σ",collect(zip(x.idx,x.val)))

function Base.:+(a::LinComb{T},b::LinComb{T}) where T
    idx = Int[]; val = T[]
    i = j = 1
    while i <= length(a.idx) || j <= length(b.idx)
        if j > length(b.idx) || (i <= length(a.idx) && a.idx[i] < b.idx[j])
            push!(idx,a.idx[i]); push!(val,a.val[i]); i += 1
        elseif i > length(a.idx) || b.idx[j] < a.idx[i]
            push!(idx,b.idx[j]); push!(val,b.val[j]); j += 1
        else
            push!(idx,a.idx[i]); push!(val,a.val[i]+b.val[j]); i += 1; j += 1
        end
    end
    LinComb{T}(a.c+b.c,idx,val)
end
Base.:-(a::LinComb) = LinComb(-a.c,a.idx,-a.val)
Base.:-(a::LinComb{T},b::LinComb{T}) where T = a + (-b)
Base.:*(a::LinComb,b::Real) = LinComb(a.c*b,a.idx,a.val*b)
Base.:*(b::Real,a::LinComb) = a*b
Base.:/(a::LinComb,b::Real) = LinComb(a.c/b,a.idx,a.val/b)
function Base.:*(a::LinComb{T},b::LinComb{T}) where T
    isconstant(a) && return b*a.c
    isconstant(b) && return a*b.c
    throw(ArgumentError("product of two parameter-dependent coefficients is not linear"))
end

evaluate(x::LinComb,θ) = x.c + sum(x.val[i]*θ[x.idx[i]] for i in eachindex(x.idx);init=zero(x.c))
evaluate(x::Real,θ) = x

#=
    Per-channel environments for channel_gradient. Growing an environment through site n: combine the bond
    environments per channel (axpys with lval or rval), one contraction per channel with the mps and the channel
    operator, then scatter onto the bond states of the next bond (axpys). DMRG itself runs on the MPSKit
    hamiltonian; these are only needed because the gradient pairs per channel.
=#
function _combine(v,idx,val)
    # idx labels the indices you need to grab from v, val labels the values you need to multiply them with
    l = rmul!(copy(v[idx[1]]),val[1])
    for i in 2:length(idx)
        l = axpy!(val[i],v[idx[i]],l)
    end
    l
end

# per channel of a site: the left environment through the channel, before its rval
function left_channel_envs(v::Vector,chs::Vector{<:Channel},A,Ab=A)
    Ab_flipped = convert(TensorMap,transpose(Ab',((1,3),(2,))))
    mapper = Map() do c
        l = _combine(v,c.lidx,c.lval)
        @planar allocator = malloc() y[-1 -2;-3] := l[4 2;1]*A[1 3;-3]*c.op[2 5;3 -2]*Ab_flipped[-1 5;4]
        y
    end
    tcollect(mapper,chs)
end

# per channel of a site: the right environment through the channel, before its lval
function right_channel_envs(v::Vector,chs::Vector{<:Channel},A,Ab=A)
    Ab_flipped = convert(TensorMap,transpose(Ab',((1,3),(2,))))
    mapper = Map() do c
        r = _combine(v,c.ridx,c.rval)
        @planar allocator = malloc() nr[-1 -2;-3] := A[-1 2;1]*r[1 3;4]*c.op[-2 5;2 3]*Ab_flipped[4 5;-3]
        nr
    end
    tcollect(mapper,chs)
end

# bond environments from the per-channel ones: lists[a] are the (channel, weight) pairs that land on bond state a;
# a state nothing lands on is an explicit zero
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

# per bond state: the (channel, weight) pairs that write to it (side = :right) or read from it (side = :left)
function _lists(chs::Vector{Channel{E,O}},nstates,side) where {E,O}
    lists = [Tuple{Int,E}[] for _ in 1:nstates]
    for (p,c) in enumerate(chs)
        idx,val = side === :right ? (c.ridx,c.rval) : (c.lidx,c.lval)
        for (a,w) in zip(idx,val)
            push!(lists[a],(p,w))
        end
    end
    lists
end

# environments on the boundary bonds (one bond state each, with virtual spaces Vl and Vr)
function boundary_environments(state::FiniteMPS,Vl,Vr)
    lll = l_LL(state);rrr = r_RR(state)
    util_left = ones(scalartype(state.AL[1]),Vl');
    @plansor ctl[-1 -2; -3]:= lll[-1;-3]*util_left[-2]
    util_right = ones(scalartype(state.AL[1]),Vr);
    @plansor ctr[-1 -2; -3]:= rrr[-1;-3]*util_right[-2]
    return ctl,ctr
end

# Pairing of a left and a right environment across the bond tensor c: E = Σ_a ⟨L[a] | R[a]⟩. Per channel slice s it
# is tr(c† L_s c R_s) = ⟨L_s† c, c R_s⟩, so both sides are contracted with c once and every pair is an inner product.
_pairleft(l,c) = (@planar A[-1 -2; -3] := conj(l[1 -2; -1]) * c[1; -3]; A)
_pairright(r,c) = (@planar B[-1 -2; -3] := c[-1; 1] * r[1 -2; -3]; B)
_pair(l,r,c) = dot(_pairleft(l,c),_pairright(r,c))

# All pairs dot(A,B) between left and right environments at once: per tensor space, one matrix product of the raw data,
# with every sector block weighted by its quantum dimension (that is what dot does).
function _pair_matrix(As::Vector{<:Pair},Bs::Vector{<:Pair})
    bygroup(xs) = (d = Dict{Any,Vector{Int}}(); for (p,(k,t)) in enumerate(xs); push!(get!(d,space(t),Int[]),p); end; d)
    ga = bygroup(As); gb = bygroup(Bs)
    G = Dict{Tuple{Any,Any},real(scalartype(last(first(As))))}()
    for (sp,pa) in ga
        pb = get(gb,sp,nothing)
        isnothing(pb) && continue
        w = copy(last(As[pa[1]]))
        for (c,b) in blocks(w); fill!(b,dim(c)); end
        Am = reduce(hcat,[w.data .* last(As[p]).data for p in pa])
        Bm = reduce(hcat,[last(Bs[p]).data for p in pb])
        M = real(Am'*Bm)
        for (ia,p) in enumerate(pa), (ib,q) in enumerate(pb)
            G[(first(As[p]),first(Bs[q]))] = M[ia,ib]
        end
    end
    G
end

"""
    channel_gradient(ψ, chs, nstates; nparams) -> (g, E)

∂⟨H(θ)⟩/∂θ for a hamiltonian given as channels whose weights are `LinComb`s in θ, pruned (`prune_channels`) so
that the boundary bonds have one state each; `ψ` must be normalized. The hamiltonian is linear in θ, so every
path through the channels carries at most one θ-dependent weight, and the derivative with respect to that weight
pairs the environment of H(θ = 0) left of it with the one right of it. The gradient is independent of θ; E is the
θ-independent part ⟨H(0)⟩.
"""
function channel_gradient(ψ::FiniteMPS,chs,nstates;nparams::Int)
    N = length(ψ)
    T = real(scalartype(ψ))
    ch0 = evaluate_channels(chs,zeros(T,nparams);tol = nothing)
    spaces = channel_bondspaces(ch0,nstates)
    (Lb,Rb) = boundary_environments(ψ,only(spaces[1]),only(spaces[N+1]))
    g = zeros(T,nparams)

    # right sweep: per site the per-channel right environments, and the bond environments of the bond right of it,
    # kept on disk like the environments
    files = [tempname() for _ in 1:N]
    try
        R = [Rb]
        for n in N:-1:1
            A = ψ.AR[n]
            rs = right_channel_envs(R,ch0[n],A)
            serialize(files[n],(rs,R))
            R = _scatter(rs,_lists(ch0[n],nstates[n],:left),a -> zeros(scalartype(A),space(A,1)*spaces[n][a]←space(A,1)))
        end

        # left sweep: a θ-dependent lval of a channel on site n pairs the left bond environment it reads with the
        # right environment through the channel, a θ-dependent rval the left environment through the channel with
        # the right bond environment it writes
        L = [Lb]
        for n in 1:N
            A = ψ.AL[n]
            (rs,Rnext) = deserialize(files[n])
            ys = left_channel_envs(L,ch0[n],A)
            _accumulate!(g,chs[n],:lval,L,rs,ψ.C[n-1])
            _accumulate!(g,chs[n],:rval,ys,Rnext,ψ.C[n])
            L = _scatter(ys,_lists(ch0[n],nstates[n+1],:right),a -> zeros(scalartype(A),space(A,3)'*spaces[n+1][a]'←space(A,3)'))
        end
        return g,real(_pair(only(L),Rb,ψ.C[N]))
    finally
        foreach(f -> rm(f; force = true),files)
    end
end

# g += Σ (∂weight/∂θ) · ⟨left | right⟩ over the θ-dependent weights of one side of the channels of a site:
# for :lval the left environments are bond states and the right ones channels, for :rval the other way around
function _accumulate!(g,chs,side,lefts,rights,c)
    need = Tuple{Int,Int}[]         # (bond state, channel) pairs
    for (k,ch) in enumerate(chs), (a,x) in (side === :lval ? zip(ch.lidx,ch.lval) : zip(ch.ridx,ch.rval))
        (x isa LinComb && !isconstant(x)) && push!(need,(a,k))
    end
    isempty(need) && return g
    states = unique(first.(need)); chans = unique(last.(need))
    G = if side === :lval
        _pair_matrix([a => _pairleft(lefts[a],c) for a in states],[k => _pairright(rights[k],c) for k in chans])
    else
        _pair_matrix([k => _pairleft(lefts[k],c) for k in chans],[a => _pairright(rights[a],c) for a in states])
    end
    for (k,ch) in enumerate(chs), (a,x) in (side === :lval ? zip(ch.lidx,ch.lval) : zip(ch.ridx,ch.rval))
        (x isa LinComb && !isconstant(x)) || continue
        Gak = get(G,side === :lval ? (a,k) : (k,a),nothing)
        isnothing(Gak) && continue
        for (q,v) in zip(x.idx,x.val)
            g[q] += Gak*v
        end
    end
    g
end
