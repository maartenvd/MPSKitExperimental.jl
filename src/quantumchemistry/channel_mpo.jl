#=
    A hamiltonian given as channels in a bond basis. Channel (lidx, lval, op, rval, ridx) on site n reads bond
    states lidx of bond n (with weights lval), applies op, and writes bond states ridx of bond n+1 (with weights
    rval): it contributes lval[a]·rval[b]·op to the MPO entry W_n[a,b]. The environments start on bond state `start`
    and end on bond state `done`; MPSKit's Jordan form needs both on every bond, passed on by identities only.

    The weights can be symbolic (LinComb, see channel_gradient.jl): the qchem builder makes its channels once per
    number of orbitals, quantum_chemistry_hamiltonian evaluates them, and qchem_rdms differentiates them.
=#
struct Channel{E,O}
    lidx::Vector{Int}
    lval::Vector{E}
    op::O
    rval::Vector{E}
    ridx::Vector{Int}
end

"""
    evaluate_channels(chs, θ; tol = 1e-12)

The channels with weights evaluated at θ. Weights with absolute value ≤ `tol` are dropped; with `tol = nothing`
every structurally present weight stays, also when it evaluates to zero.
"""
function evaluate_channels(chs,θ;tol = 1e-12)
    map(chs) do site
        map(site) do c
            lv = [evaluate(x,θ) for x in c.lval]; rv = [evaluate(x,θ) for x in c.rval]
            kl = isnothing(tol) ? trues(length(lv)) : abs.(lv) .> tol
            kr = isnothing(tol) ? trues(length(rv)) : abs.(rv) .> tol
            Channel(c.lidx[kl],lv[kl],c.op,rv[kr],c.ridx[kr])
        end
    end
end

"""
    prune_channels(chs, nstates; start = 1, done = nstates[end]) -> (chs, kept)

Keeps the bond states and channels that lie on a path from `start` on the first bond to `done` on the last one,
and renumbers the bond states: `kept[b]` are the old numbers of the states that remain on bond `b`.
"""
function prune_channels(chs,nstates;start::Int = 1,done::Int = nstates[end])
    N = length(chs)
    # reachable from the left boundary state, and from the right one
    fwd = [falses(nstates[b]) for b in 1:N+1]; fwd[1][start] = true
    for n in 1:N, c in chs[n]
        any(fwd[n][c.lidx]) && (fwd[n+1][c.ridx] .= true)
    end
    bwd = [falses(nstates[b]) for b in 1:N+1]; bwd[N+1][done] = true
    for n in N:-1:1, c in chs[n]
        any(bwd[n+1][c.ridx]) && (bwd[n][c.lidx] .= true)
    end
    live = [fwd[b] .& bwd[b] for b in 1:N+1]
    newidx = [cumsum(l) .* l for l in live]
    out = map(1:N) do n
        site = eltype(chs[n])[]
        for c in chs[n]
            kl = live[n][c.lidx]; kr = live[n+1][c.ridx]
            (any(kl) && any(kr)) || continue
            push!(site,Channel(newidx[n][c.lidx[kl]],c.lval[kl],c.op,c.rval[kr],newidx[n+1][c.ridx[kr]]))
        end
        site
    end
    return out,[findall(l) for l in live]
end

"""
    split_sectors(chs, nstates) -> (chs, nstates, parts)

The same hamiltonian with every bond state that carries several sectors split into one bond state per sector:
the channels are projected onto every (left sector, right sector) combination of their operator's virtual legs.
`parts[b]` lists, per new bond state of bond `b`, the old state and the sector. Nothing changes in cost (the
operators were block diagonal in these sectors anyway), but every bond state now has a single sector, which
is what pairing bond states with their hermitian conjugates needs.
"""
function split_sectors(chs,nstates)
    N = length(chs)
    spaces = channel_bondspaces(chs,nstates)
    parts = [[(a,s) for a in 1:nstates[b] for s in sectors(spaces[b][a])] for b in 1:N+1]
    newidx = [Dict(p => i for (i,p) in enumerate(parts[b])) for b in 1:N+1]
    restrict(V,s) = isdual(V) ? typeof(V)(dual(s) => dim(V,s))' : typeof(V)(s => dim(V,s))   # the sector-s part of V
    out = map(1:N) do n
        site = eltype(chs[n])[]
        for c in chs[n]
            Vl = space(c.op,1); Vr = space(c.op,4)'          # the bond spaces it reads and writes (not dual)
            for s in sectors(Vl), t in sectors(Vr)
                El = isometry(storagetype(c.op),Vl,restrict(Vl,s))
                Er = isometry(storagetype(c.op),Vr,restrict(Vr,t))
                @planar o[-1 -2; -3 -4] := El'[-1; 1] * c.op[1 -2; -3 2] * Er[2; -4]
                norm(o) < 1e-14*max(norm(c.op),1) && continue
                push!(site,Channel([newidx[n][(a,s)] for a in c.lidx],c.lval,o,c.rval,[newidx[n+1][(b,t)] for b in c.ridx]))
            end
        end
        site
    end
    return out,length.(parts),parts
end

# per bond state: its virtual space, from the channels that read it (space(op,1)) or write it (space(op,4)')
function channel_bondspaces(chs,nstates)
    N = length(chs)
    S = typeof(space(first(first(chs)).op,1))
    sp = [Vector{Union{Nothing,S}}(nothing,nstates[b]) for b in 1:N+1]
    function set!(b,a,V)
        isnothing(sp[b][a]) ? (sp[b][a] = V) : (sp[b][a] == V || throw(SpaceMismatch("bond state $a of bond $b")))
    end
    for n in 1:N, c in chs[n]
        foreach(a -> set!(n,a,space(c.op,1)),c.lidx)
        foreach(b -> set!(n+1,b,space(c.op,4)'),c.ridx)
    end
    any(b -> any(isnothing,b),sp) && throw(ArgumentError("bond state without channels"))
    return [Vector{S}(b) for b in sp]
end

"""
    channel_hamiltonian(chs, nstates; start, done) -> FiniteMPOHamiltonian

The MPSKit hamiltonian with W_n[a,b] = Σ lval[a]·rval[b]·op over the channels of site `n`, on the given bond basis.
`start[b]`/`done[b]` are the start and done states of bond `b`; MPSKit's Jordan form puts them first and last.
"""
function channel_hamiltonian(chs,nstates;start::Vector{Int},done::Vector{Int})
    N = length(chs)
    spaces = channel_bondspaces(chs,nstates)
    λs = [[_identity_coefficient(c.op) for c in site] for site in chs]
    for n in 1:N-1
        _check_passthrough(chs[n],λs[n],start[n],start[n+1],:start) || throw(ArgumentError("no Jordan form: start state of bond $(n+1)"))
    end
    for n in 2:N
        _check_passthrough(chs[n],λs[n],done[n],done[n+1],:done) || throw(ArgumentError("no Jordan form: done state of bond $n"))
    end
    perm = map(1:N+1) do b
        nstates[b] == 1 && return [1]
        start[b] != done[b] || throw(ArgumentError("no Jordan form: start and done coincide on bond $b"))
        [start[b]; [a for a in 1:nstates[b] if a != start[b] && a != done[b]]; done[b]]
    end
    position = [invperm(p) for p in perm]
    return FiniteMPOHamiltonian(map(1:N) do n
        _jordan_mpotensor(chs[n],λs[n],spaces[n][perm[n]],spaces[n+1][perm[n+1]],position[n],position[n+1])
    end)
end

# the start state of the next bond is written only by identities that read only the start state of this bond
# (and the done state of this bond is read only by identities that write only to the done state of the next one)
function _check_passthrough(site,λs,a,b,which)
    for (c,λ) in zip(site,λs)
        touches = which === :start ? b in c.ridx : a in c.lidx
        touches || continue
        isnothing(λ) && return false
        (which === :start ? c.lidx == [a] : c.ridx == [b]) || return false
    end
    return true
end

# the identity channel operator on (v ⊗ p ← p ⊗ v), built from TensorKit alone (MPSKit's similar_braidingtensor
# is not in every MPSKit version this package runs with)
function _identity_operator(e)
    virt = isomorphism(storagetype(e),space(e,1),space(e,1))
    phys = isomorphism(storagetype(e),space(e,2),space(e,2))
    @plansor id_e[-1 -2;-3 -4] := virt[-1;1]*phys[-2;2]*τ[1 2;-3 -4]
    id_e
end

function _identity_coefficient(e)
    space(e,1) == space(e,4)' || return nothing
    id_e = _identity_operator(e)
    space(id_e) == space(e) || return nothing
    λ = dot(id_e,e)/dot(id_e,id_e)
    return norm(e - λ*id_e) <= 1.0e-12*max(norm(e),1) ? λ : nothing
end

function _jordan_mpotensor(site,λs,Vls,Vrs,lpos,rpos)
    E = promote_type(map(c -> promote_type(eltype(c.lval),eltype(c.rval)),site)...)
    P = space(first(site).op,2)
    W = MPSKit.jordanmpotensortype(spacetype(P),Vector{E})(undef,MPSKit.SumSpace(Vls) ⊗ P ← P ⊗ MPSKit.SumSpace(Vrs))

    tensors = Dict{Tuple{Int,Int},Any}()
    scalars = Dict{Tuple{Int,Int},E}()
    for (c,λ) in zip(site,λs)
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
        axpy!(s,_identity_operator(tensors[key]),tensors[key])
        delete!(scalars,key)
    end

    for ((a,b),t) in tensors
        W[a,1,1,b] = t
    end
    for ((a,b),s) in scalars
        _setscalar!(W,s,a,b)
    end
    return W
end

# MPSKit versions differ here: newer JordanMPOTensors store identity entries as scalars, older ones (A/B/C/D
# blocks) only take tensors and keep the identity corners implicit
function _setscalar!(W,s,a,b)
    if hasproperty(W,:scalars)
        W[a,1,1,b] = s
    elseif (a,b) == (1,1) && size(W,4) > 1 || (a,b) == (size(W,1),size(W,4)) && size(W,1) > 1
        isapprox(s,1) || throw(ArgumentError("identity corner ($a,$b) with coefficient $s"))
    else
        W[a,1,1,b] = s*_identity_operator(zeros(scalartype(W),MPSKit.eachspace(W)[a,1,1,b]))
    end
    W
end
