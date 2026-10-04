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

# Σ_i val[i] * v[idx[i]]
function _combine(v,idx,val)
    # idx labels the indices you need to grab from v, val labels the values you need to multiply them with
    l = rmul!(copy(v[idx[1]]),val[1])
    for i in 2:length(idx)
        l = axpy!(val[i],v[idx[i]],l)
    end
    l
end

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

#=
    The gradient of ⟨H(θ)⟩ with respect to θ, for a hamiltonian given as channels whose weights are LinComb's in θ.
    H is linear in θ, so every path through the channels carries at most one θ-dependent weight. For a weight w of
    channel k on site n,

        ∂E/∂w = ⟨AC_n| L ⊗ op_k ⊗ R |AC_n⟩,

    with L the left bond environment the weight sits on (or the channel's weighted sum of them) and R the right one,
    all environments of H(θ = 0) with every structural weight kept. So per site the bond environments are contracted
    with AC once per bond state (U_a = L_a·AC, W_b = R_b·conj(AC)), every channel applies its operator once to a
    weighted sum of those, and every θ-dependent weight is one small contraction. The bond environments are
    MPSKit's own, on disk.
=#
"""
    GradientStructure(chs, nstates; start, done, nparams)
    channel_gradient(ψ, S::GradientStructure) -> (g, E)

`chs` the pruned channels with LinComb weights in θ (`nparams` parameters), on bond states that start on `start`
and end on `done`. Returns ∂⟨H(θ)⟩/∂θ (independent of θ) and ⟨H(0)⟩; `ψ` must be normalized.
"""
struct GradientStructure{C0,C,H}
    chs::C                  # per site, the channels with symbolic (LinComb) weights, pruned
    ch0::C0                 # the same with the weights at θ = 0, every structural weight kept
    H0::H                   # FiniteMPOHamiltonian of ch0
    position::Vector{Vector{Int}}   # per bond: the position of every bond state in H0's (Jordan) order
    nparams::Int
end
function GradientStructure(chs,nstates;start,done,nparams)
    ch0 = evaluate_channels(chs,zeros(nparams);tol = nothing)
    H0 = channel_hamiltonian(ch0,nstates;start,done)
    GradientStructure(chs,ch0,H0,[invperm(p) for p in _jordan_perm(nstates,start,done)],nparams)
end

_isθ(x) = x isa LinComb && !isconstant(x)

function channel_gradient(ψ::FiniteMPS,S::GradientStructure)
    N = length(ψ)
    T = real(scalartype(ψ))
    g = zeros(T,S.nparams)
    envs = disk_environments(ψ,S.H0)
    for n in 1:N
        GL = leftenv(envs,n,ψ); GR = rightenv(envs,n,ψ)
        AC = ψ.AC[n]
        pl = S.position[n]; pr = S.position[n+1]
        sym = S.chs[n]; num = S.ch0[n]
        # which bond states this site needs
        needU = Set{Int}(); needW = Set{Int}()
        for (cs,c) in zip(sym,num)
            θl = any(_isθ,cs.lval); θr = any(_isθ,cs.rval)
            (θl || θr) || continue
            θr && union!(needU,c.lidx)
            θl && (union!(needU,c.lidx[_isθ.(cs.lval)]); union!(needW,c.ridx))
            θr && union!(needW,c.ridx[_isθ.(cs.rval)])
        end
        U = Dict(a => (@planar u[-1 -2 -3; -4] := GL[1,pl[a],1][-1 -2; 1]*AC[1 -3; -4]; u) for a in needU)
        W = Dict(b => (@planar w[-1 -2; -3 -4] := GR[1,pr[b],1][-1 -2; 1]*conj(AC[-3 -4; 1]); w) for b in needW)
        # every pairing written as an inner product dot(A, B), so that all of a site's pairings are a few matrix
        # products (_pair_matrix): θ-dependent rval of channel k on state b pairs A_b = conj(W_b), laid out like U,
        # with Y_k = (Σ_a lval U_a)·op_k; θ-dependent lval on state a pairs U_a with Ã_k = conj(op_k·Σ_b rval W_b)
        Ys = Pair{Int,Any}[]; Atil = Pair{Int,Any}[]
        for (k,(cs,c)) in enumerate(zip(sym,num))
            if any(_isθ,cs.rval)
                Us = _combine(U,c.lidx,c.lval)
                @planar Y[-1 -2 -3; -4] := Us[-1 1 2; -4]*c.op[1 -2; 2 -3]
                push!(Ys,k => Y)
            end
            if any(_isθ,cs.lval)
                Ws = _combine(W,c.ridx,c.rval)
                @planar At[-1 -2 -3; -4] := conj(c.op[-2 5; -3 6])*conj(Ws[-4 6; -1 5])
                push!(Atil,k => At)
            end
        end
        if !isempty(Ys)
            θb = unique([b for (k,_) in Ys for (b,x) in zip(sym[k].ridx,sym[k].rval) if _isθ(x)])
            G = _pair_matrix([b => (@planar A[-1 -2 -3; -4] := conj(W[b][-4 -3; -1 -2])) for b in θb],Ys)
            for (k,_) in Ys, (b,x) in zip(sym[k].ridx,sym[k].rval)
                _isθ(x) || continue
                Gbk = G[(b,k)]
                for (q,v) in zip(x.idx,x.val); g[q] += Gbk*v; end
            end
        end
        if !isempty(Atil)
            G = _pair_matrix(Atil,[a => U[a] for a in unique([a for (k,_) in Atil for (a,x) in zip(sym[k].lidx,sym[k].lval) if _isθ(x)])])
            for (k,_) in Atil, (a,x) in zip(sym[k].lidx,sym[k].lval)
                _isθ(x) || continue
                Gka = G[(k,a)]
                for (q,v) in zip(x.idx,x.val); g[q] += Gka*v; end
            end
        end
    end
    return g,real(expectation_value(ψ,S.H0,envs))
end
