"""
    FiniteMPOHamiltonian(ham::LinkMPOHamiltonian)

The same operator as a regular MPSKit `FiniteMPOHamiltonian` on exactly the same bond states (the environment
basis of `ham`), so that MPSKit's own `JordanMPOTensor` code paths can be compared against the link ones.

The site tensor is W_n[a,b] = Σ_k Y_n[a,k] O_k X_{n+1}[k,b]: every channel contributes `lval[a] * rval[b] * op`
to the entry `(a, b)`, and channels whose operator is a multiple of the identity become identity scalars.
MPSKit's Jordan form [1 C D; 0 A B; 0 0 1] needs the bond state that only passes the identity on from the left
boundary ("start") first and the one that only passes it on to the right boundary ("done") last, so those are found
and moved there.
"""
function MPSKit.FiniteMPOHamiltonian(ham::LinkMPOHamiltonian)
    N = length(ham)
    λs = [[_identity_coefficient(c.op) for c in ham.channels[n]] for n in 1:N]
    nstates = length.(ham.bondspaces)

    # start: written only by identity channels that read only the start state of the previous bond
    start = zeros(Int,N+1); start[1] = 1
    for n in 1:N-1
        writers = [Int[] for _ in 1:nstates[n+1]]
        for (p,c) in enumerate(ham.channels[n]), a in c.ridx; push!(writers[a],p); end
        cands = findall(eachindex(writers)) do a
            !isempty(writers[a]) &&
                all(p -> !isnothing(λs[n][p]) && ham.channels[n][p].lidx == [start[n]],writers[a])
        end
        length(cands) == 1 || throw(ArgumentError("no Jordan form: $(length(cands)) start states on bond $(n+1)"))
        start[n+1] = only(cands)
    end

    # done: read only by identity channels that write only to the done state of the next bond
    done = zeros(Int,N+1); done[N+1] = 1
    for n in N:-1:2
        readers = [Int[] for _ in 1:nstates[n]]
        for (p,c) in enumerate(ham.channels[n]), a in c.lidx; push!(readers[a],p); end
        cands = findall(eachindex(readers)) do a
            !isempty(readers[a]) &&
                all(p -> !isnothing(λs[n][p]) && ham.channels[n][p].ridx == [done[n+1]],readers[a])
        end
        length(cands) == 1 || throw(ArgumentError("no Jordan form: $(length(cands)) done states on bond $n"))
        done[n] = only(cands)
    end

    # bond n in Jordan order: start, the rest, done (the boundary bonds have a single state)
    perm = map(1:N+1) do n
        nstates[n] == 1 && return [1]
        start[n] != done[n] || throw(ArgumentError("no Jordan form: start and done coincide on bond $n"))
        [start[n]; [a for a in 1:nstates[n] if a != start[n] && a != done[n]]; done[n]]
    end
    position = [invperm(p) for p in perm]

    return FiniteMPOHamiltonian(map(1:N) do n
        _jordan_mpotensor(ham,n,λs[n],perm[n],perm[n+1],position[n],position[n+1])
    end)
end

function _identity_coefficient(e)
    space(e,1) == space(e,4)' || return nothing
    id_e = TensorMap(MPSKit.similar_braidingtensor(e))
    λ = dot(id_e,e)/dot(id_e,id_e)
    return norm(e - λ*id_e) <= 1.0e-12*max(norm(e),1) ? λ : nothing
end

function _jordan_mpotensor(ham::LinkMPOHamiltonian{E},n,λs,lperm,rperm,lpos,rpos) where {E}
    P = ham.pspaces[n]
    Vl = MPSKit.SumSpace(ham.bondspaces[n][lperm])
    Vr = MPSKit.SumSpace(ham.bondspaces[n+1][rperm])
    W = MPSKit.jordanmpotensortype(spacetype(P),Vector{E})(undef,Vl ⊗ P ← P ⊗ Vr)

    tensors = Dict{Tuple{Int,Int},Any}()
    scalars = Dict{Tuple{Int,Int},E}()
    for (c,λ) in zip(ham.channels[n],λs)
        for (a,la) in zip(c.lidx,c.lval), (b,rb) in zip(c.ridx,c.rval)
            coef = la*rb
            iszero(coef) && continue
            key = (lpos[a],rpos[b])
            if isnothing(λ)
                tensors[key] = haskey(tensors,key) ? axpy!(coef,c.op,tensors[key]) : coef*c.op
            else
                scalars[key] = get(scalars,key,zero(E)) + coef*λ
            end
        end
    end
    # an entry with both kinds of contribution has to be stored as a single tensor
    for (key,s) in collect(scalars)
        haskey(tensors,key) || continue
        axpy!(s,TensorMap(MPSKit.similar_braidingtensor(tensors[key])),tensors[key])
        delete!(scalars,key)
    end

    for ((a,b),t) in tensors
        W[a,1,1,b] = t
    end
    for ((a,b),s) in scalars
        W[a,1,1,b] = s
    end
    return W
end
