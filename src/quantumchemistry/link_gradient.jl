#=
    In quantum chemistry (and actually generically, noone is stopping you) we're able to put all degrees of freedom on the link matrices.
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

"""
    evaluate(links, θ; tol = 1e-12) -> links with numbers

Entries with absolute value ≤ `tol` are dropped; with `tol = nothing` every structurally present entry stays
stored, also when it evaluates to zero.
"""
function evaluate(links::Vector{<:SparseMatrixCSC{<:LinComb{T}}},θ;tol = 1e-12) where T
    map(links) do C
        I,J,V = findnz(C)
        out = sparse(I,J,T[evaluate(v,θ) for v in V],size(C)...)
        isnothing(tol) ? out : droptol!(out,tol)
    end
end

# Pairing of a left and a right environment across the bond tensor c: E = Σ_a ⟨L[a] | R[a]⟩. Per channel slice s it
# is tr(c† L_s c R_s) = ⟨L_s† c, c R_s⟩, so both sides are contracted with c once and every pair is an inner product.
_pairleft(l,c) = (@planar A[-1 -2; -3] := conj(l[1 -2; -1]) * c[1; -3]; A)
_pairright(r,c) = (@planar B[-1 -2; -3] := c[-1; 1] * r[1 -2; -3]; B)
_pair(l,r,c) = dot(_pairleft(l,c),_pairright(r,c))

# All pairs dot(A,B) between left and right channels at once: per tensor space, one matrix product of the raw data,
# with every sector block weighted by its quantum dimension (that is what dot does).
function _pair_matrix(As::Vector{<:Pair},Bs::Vector{<:Pair})
    bygroup(xs) = (d = Dict{Any,Vector{Int}}(); for (p,(k,t)) in enumerate(xs); push!(get!(d,space(t),Int[]),p); end; d)
    ga = bygroup(As); gb = bygroup(Bs)
    G = Dict{Tuple{Int,Int},real(scalartype(last(first(As))))}()
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
    gradient_hamiltonian(ops, links, pspaces) -> LinkMPOHamiltonian

H(θ = 0) with the full sparsity pattern of the `LinComb` links, so every entry that carries a parameter keeps
its channels. It does not depend on θ; `link_gradient` needs its environments.
"""
function gradient_hamiltonian(ops,links::Vector{<:SparseMatrixCSC{LinComb{T}}},pspaces) where T
    np = maximum(C -> maximum(x -> isempty(x.idx) ? 0 : x.idx[end],nonzeros(C);init=0),links)
    LinkMPOHamiltonian(ops,evaluate(links,zeros(T,np);tol=nothing),pspaces;dropzeros=false)
end

"""
    link_gradient(ψ, h, links; nparams) -> (g, E)

`links` with `LinComb` entries and `h = gradient_hamiltonian(ops, links, pspaces)`. Returns ∂⟨H(θ)⟩/∂θ
(independent of θ for a hamiltonian linear in θ) and the θ-independent part E = ⟨H(0)⟩. `ψ` must be normalized.
"""
function link_gradient(ψ::FiniteMPS,h::LinkMPOHamiltonian,links::Vector{<:SparseMatrixCSC{LinComb{T}}};
                       nparams::Int = maximum(C -> maximum(x -> isempty(x.idx) ? 0 : x.idx[end],nonzeros(C);init=0),links)) where T
    N = length(ψ)
    (Lb,Rb) = boundary_environments(ψ,h)
    g = zeros(T,nparams)

    # right sweep: the per-channel right environments of every site, kept on disk like the environments
    files = [tempname() for _ in 1:N]
    try
        R = [Rb]
        for n in N:-1:1
            rs = right_channel_envs(R,h,n,ψ.AR[n])
            serialize(files[n],rs)
            states = h.bondspaces[n]
            R = _scatter(rs,_lists(h.channels[n],length(states),:left),
                         a -> zeros(scalartype(ψ.AR[n]),space(ψ.AR[n],1)*states[a]←space(ψ.AR[n],1)))
        end

        # left sweep: pair across every link, then grow the left environment with the same per-channel results
        L = [Lb]
        E = zero(T)
        for n in 1:N+1
            if n == 1
                ys = Dict(1 => Lb)
            else
                yv = left_channel_envs(L,h,n-1,ψ.AL[n-1])
                ys = Dict(c.k => y for (c,y) in zip(h.channels[n-1],yv))
                states = h.bondspaces[n]
                A = ψ.AL[n-1]
                L = _scatter(yv,_lists(h.channels[n-1],length(states),:right),
                             a -> zeros(scalartype(A),space(A,3)'*states[a]'←space(A,3)'))
            end
            rs = n == N+1 ? Dict(1 => Rb) : Dict(c.k => r for (c,r) in zip(h.channels[n],deserialize(files[n])))
            c = ψ.C[n-1]
            n == N+1 && (E = real(_pair(only(L),Rb,c)))

            C = links[n]
            any(x -> !isconstant(x),nonzeros(C)) || continue
            G = _pair_matrix([k => _pairleft(y,c) for (k,y) in ys],[k => _pairright(r,c) for (k,r) in rs])
            rows = rowvals(C); vals = nonzeros(C)
            for j in 1:size(C,2), p in nzrange(C,j)
                x = vals[p]
                isconstant(x) && continue
                Gij = get(G,(rows[p],j),nothing)
                isnothing(Gij) && continue
                for (q,v) in zip(x.idx,x.val)
                    g[q] += Gij*v
                end
            end
        end
        return g, E
    finally
        foreach(f -> rm(f; force = true),files)
    end
end
