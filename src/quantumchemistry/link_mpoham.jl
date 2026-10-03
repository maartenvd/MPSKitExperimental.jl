#=
    A hamiltonian as operators on the sites and scalar matrices on the links,

        H = b_L · diag(O_1) · C_1 · diag(O_2) · C_2 ⋯ C_{N-1} · diag(O_N) · b_R.

    ops[n][k] is the operator of channel k on site n (legs: left virtual ⊗ phys ← phys ⊗ right virtual).
    links[n] is the link in front of site n, a K_{n-1} × K_n matrix; links[1] = b_L is 1 × K_1 and
    links[N+1] = b_R is K_N × 1.

    Environments can now be stored on the inner index of a factorization of the link matrix.
    Here we automatically derive automatically a decent factorization, you just need to supply links and ops.

    This is a construction format: DMRG runs on FiniteMPOHamiltonian(h) (jordan_conversion.jl), which uses this
    factorization as its bond basis. On that basis MPSKit's transfers are already optimal (A and Ā once per bond
    state is a minimum vertex cover of the channel graphs), so the link format only gained a constant factor at
    the NC/CN switch. The links stay useful for building and for link_gradient (RDMs).
=#

#=
lidx keeps track of "which entries in the environment should I grab
lval keeps track of "I grabbed them, which scalar values do I multiply them with?"
O is the operator that needs to be applied
rval keeps track of "great, you did the contraction, now create a copy for each entry with rval, and multiply with the scalar rvals"
ridx then tells us where each of those results need to be added to

Obviously you can read it also the other way around.
=#
struct LinkChannel{E,O}
    k::Int                  # index into ops[n]
    lidx::Vector{Int}
    lval::Vector{E}
    op::O
    rval::Vector{E}
    ridx::Vector{Int}
end

struct LinkMPOHamiltonian{E,O,Sp}
    ops::Vector{Vector{O}}
    links::Vector{SparseMatrixCSC{E,Int}}
    pspaces::Vector{Sp}

    # automatically derived
    X::Vector{SparseMatrixCSC{E,Int}}
    Y::Vector{SparseMatrixCSC{E,Int}}
    bondspaces::Vector{Vector{Sp}}      # bondspaces[n][a]: bond state a of link n, as the left virtual space of site n
    channels::Vector{Vector{LinkChannel{E,O}}}
end


# ---- I have not read beyond this point. LinkMPOHamiltonian is of the structure that I used to have, the automatic construction I don't yet understand ----
"""
    LinkMPOHamiltonian(ops, links, pspaces; dropzeros = true)

`ops[n]` the channel operators of site `n`, `links[n]` the scalar matrix between the channels of site `n-1` and `n`
(`links[1]` and `links[end]` the boundary vectors). With `dropzeros = false` explicitly stored zeros keep their
place in the sparsity pattern.
"""
function LinkMPOHamiltonian(ops::Vector{Vector{O}},links::Vector{SparseMatrixCSC{E,Int}},pspaces::Vector{Sp};
                            dropzeros::Bool = true) where {E,O,Sp}
    N = length(ops)
    length(links) == N+1 || throw(ArgumentError("need one link more than sites"))
    for n in 1:N
        size(links[n],2) == length(ops[n]) == size(links[n+1],1) ||
            throw(DimensionMismatch("links around site $n don't match its $(length(ops[n])) channels"))
    end
    size(links[1],1) == 1 && size(links[end],2) == 1 || throw(ArgumentError("boundary links must be vectors"))
    links = dropzeros ? map(SparseArrays.dropzeros,links) : links
    links = _prune_dead_channels(links)

    XY = map(1:N+1) do n
        n == 1   && return (sparse([1],[1],[one(E)],1,1),links[1],[(:row,1)])
        n == N+1 && return (links[N+1],sparse([1],[1],[one(E)],1,1),[(:col,1)])
        _star_factor(links[n])
    end
    X = [xy[1] for xy in XY]; Y = [xy[2] for xy in XY]

    # a bond state takes the virtual space of its hub channel
    stored(v) = rowvals(v)[nzrange(v,1)]
    bondspaces = map(1:N+1) do n
        map(XY[n][3]) do (side,k)
            sp = if n == 1
                unique(space(ops[1][j],1) for j in stored(sparse(transpose(links[1]))))
            elseif n == N+1
                unique(space(ops[N][i],4)' for i in stored(links[N+1]))
            else
                [side === :row ? space(ops[n-1][k],4)' : space(ops[n][k],1)]
            end
            length(sp) == 1 || throw(SpaceMismatch("boundary channels have different virtual spaces"))
            only(sp)
        end
    end

    channels = map(1:N) do n
        Xt = sparse(transpose(X[n+1]))     # column k of Y[n] and of Xt: channel k
        chs = LinkChannel{E,O}[]
        for k in eachindex(ops[n])
            lp = nzrange(Y[n],k); rp = nzrange(Xt,k)
            (isempty(lp) || isempty(rp)) && continue
            push!(chs,LinkChannel{E,O}(k,rowvals(Y[n])[lp],nonzeros(Y[n])[lp],ops[n][k],nonzeros(Xt)[rp],rowvals(Xt)[rp]))
        end
        for c in chs
            all(==(space(c.op,1)),bondspaces[n][c.lidx]) || throw(SpaceMismatch("channel $(c.k) on site $n"))
            all(==(space(c.op,4)'),bondspaces[n+1][c.ridx]) || throw(SpaceMismatch("channel $(c.k) on site $n"))
        end
        chs
    end

    return LinkMPOHamiltonian{E,O,Sp}(ops,links,pspaces,X,Y,bondspaces,channels)
end

Base.length(h::LinkMPOHamiltonian) = length(h.ops)
MPSKit.physicalspace(h::LinkMPOHamiltonian,i::Int) = h.pspaces[i]
Base.eltype(::LinkMPOHamiltonian{E}) where E = E
Base.show(io::IO,h::LinkMPOHamiltonian) =
    print(io,"LinkMPOHamiltonian on $(length(h)) sites: $(sum(length,h.ops)) channels, $(sum(nnz,h.links)) link entries, ",
          "$(sum(length,h.bondspaces)) bond states")

"""
    link_channels(sitechannels, nstates; lstart = 1, rend = nstates[end]) -> (ops, links)

Links from a hamiltonian given in some bond basis: `sitechannels[n]` lists the channels of site `n` as
`(lidx, lval, op, rval, ridx)`, reading from bond states `lidx` of bond `n` and writing to bond states `ridx` of
bond `n+1`; `nstates[n]` is the number of bond states on bond `n`. The environments start on bond state `lstart`
of the left boundary and end on `rend` of the right one. Works for any scalar type (also `LinComb`).
"""
function link_channels(sitechannels::AbstractVector,nstates::AbstractVector{Int}; lstart::Int = 1, rend::Int = nstates[end])
    N = length(sitechannels)
    E = mapreduce(ch -> mapreduce(c -> promote_type(eltype(c[2]),eltype(c[4])),promote_type,ch),promote_type,sitechannels)
    ops = [[c[3] for c in ch] for ch in sitechannels]
    # bond n: who writes to state a (site n-1) and who reads from it (site n)
    writers(n) = (w = [Tuple{Int,E}[] for _ in 1:nstates[n]];
                  for (k,c) in enumerate(sitechannels[n-1]), (a,v) in zip(c[5],c[4]); push!(w[a],(k,v)); end; w)
    readers(n) = (r = [Tuple{Int,E}[] for _ in 1:nstates[n]];
                  for (k,c) in enumerate(sitechannels[n]), (a,v) in zip(c[1],c[2]); push!(r[a],(k,v)); end; r)
    function link(n)
        acc = Dict{Tuple{Int,Int},E}()
        if n == 1
            for (k,v) in readers(1)[lstart]; acc[(1,k)] = get(acc,(1,k),zero(E)) + v; end
            m,l = 1,length(ops[1])
        elseif n == N+1
            for (k,v) in writers(N+1)[rend]; acc[(k,1)] = get(acc,(k,1),zero(E)) + v; end
            m,l = length(ops[N]),1
        else
            w = writers(n); r = readers(n)
            for a in 1:nstates[n], (k,x) in w[a], (j,y) in r[a]
                acc[(k,j)] = get(acc,(k,j),zero(E)) + x*y
            end
            m,l = length(ops[n-1]),length(ops[n])
        end
        ks = collect(keys(acc))
        sparse(first.(ks),last.(ks),E[acc[k] for k in ks],m,l)
    end
    return ops,SparseMatrixCSC{E,Int}[link(n) for n in 1:N+1]
end

# A channel without stored incoming or outgoing entries never contributes; removing its other entries can kill
# further channels, so repeat until nothing changes. Explicitly stored zeros count as entries.
function _prune_dead_channels(links::Vector{<:SparseMatrixCSC})
    links = copy(links)
    N = length(links)-1
    while true
        changed = false
        for n in 1:N
            hasin = [!isempty(nzrange(links[n],k)) for k in 1:size(links[n],2)]
            hasout = falses(size(links[n+1],1)); hasout[rowvals(links[n+1])] .= true
            dead = .!(hasin .& hasout)
            any(dead) || continue
            keep = findall(.!dead)
            # drop column k of links[n] and row k of links[n+1] for dead k, keeping the dimensions
            if any(k -> dead[k] && !isempty(nzrange(links[n],k)),eachindex(dead)) ||
               any(i -> dead[i],rowvals(links[n+1]))
                changed = true
                I,J,V = findnz(links[n]); m = .!dead[J]
                links[n] = sparse(I[m],J[m],V[m],size(links[n])...)
                I,J,V = findnz(links[n+1]); m = .!dead[I]
                links[n+1] = sparse(I[m],J[m],V[m],size(links[n+1])...)
            end
        end
        changed || return links
    end
end

# Minimum vertex cover of the bipartite graph rows × cols given by the stored entries of C (König).
function _min_vertex_cover(C::SparseMatrixCSC)
    m,n = size(C)
    rows = rowvals(C)
    adj = [Int[] for _ in 1:m]          # row -> cols
    for j in 1:n, p in nzrange(C,j)
        push!(adj[rows[p]],j)
    end
    matchrow = zeros(Int,m); matchcol = zeros(Int,n)
    for i in 1:m, j in adj[i]           # greedy start
        if matchcol[j] == 0
            matchrow[i] = j; matchcol[j] = i
            break
        end
    end
    # Kuhn's augmenting paths. A failed search leaves the matching unchanged, so what it visited
    # stays useless until the next augmentation: only reset `seen` after a success.
    seen = falses(n)
    parent = zeros(Int,n)
    for i0 in 1:m
        (matchrow[i0] != 0 || isempty(adj[i0])) && continue
        stack = [(i0,1)]
        found = 0
        while !isempty(stack) && found == 0
            (i,p) = pop!(stack)
            p > length(adj[i]) && continue
            push!(stack,(i,p+1))
            j = adj[i][p]
            seen[j] && continue
            seen[j] = true
            parent[j] = i
            if matchcol[j] == 0
                found = j
            else
                push!(stack,(matchcol[j],1))
            end
        end
        found == 0 && continue
        j = found
        while j != 0                    # flip the path back to i0
            i = parent[j]
            nextj = matchrow[i]
            matchrow[i] = j; matchcol[j] = i
            j = i == i0 ? 0 : nextj
        end
        fill!(seen,false)
    end
    # König: Z = reachable from unmatched rows by alternating paths; cover = (rows ∖ Z) ∪ (cols ∩ Z)
    zrow = falses(m); zcol = falses(n)
    queue = [i for i in 1:m if matchrow[i] == 0 && !isempty(adj[i])]
    zrow[queue] .= true
    while !isempty(queue)
        i = pop!(queue)
        for j in adj[i]
            zcol[j] && continue
            zcol[j] = true
            i2 = matchcol[j]
            if i2 != 0 && !zrow[i2]
                zrow[i2] = true; push!(queue,i2)
            end
        end
    end
    rowcover = [!zrow[i] && !isempty(adj[i]) for i in 1:m]
    return rowcover, collect(zcol)
end

# Star decomposition C = X·Y: a covered row i becomes a bond state with X[i,a] = 1, Y[a,:] = its entries;
# a covered column j takes the remaining entries of that column, X[:,a] = those entries, Y[a,j] = 1.
function _star_factor(C::SparseMatrixCSC{E}) where E
    m,n = size(C)
    rowcover,colcover = _min_vertex_cover(C)
    rows = rowvals(C); vals = nonzeros(C)
    rowstate = cumsum(rowcover) .* rowcover
    colstate = zeros(Int,n)
    XI = Int[]; XJ = Int[]; XV = E[]; YI = Int[]; YJ = Int[]; YV = E[]
    hubs = Tuple{Symbol,Int}[(:row,i) for i in 1:m if rowcover[i]]
    for i in 1:m
        rowcover[i] || continue
        push!(XI,i); push!(XJ,rowstate[i]); push!(XV,one(E))
    end
    for j in 1:n, p in nzrange(C,j)
        i = rows[p]
        if rowcover[i]
            push!(YI,rowstate[i]); push!(YJ,j); push!(YV,vals[p])
        else
            colcover[j] || error("not a vertex cover")
            if colstate[j] == 0
                push!(hubs,(:col,j)); colstate[j] = length(hubs)
                push!(YI,colstate[j]); push!(YJ,j); push!(YV,one(E))
            end
            push!(XI,i); push!(XJ,colstate[j]); push!(XV,vals[p])
        end
    end
    A = length(hubs)
    # sparse() keeps explicitly stored zeros, which matters for dropzeros = false
    return sparse(XI,XJ,XV,m,A), sparse(YI,YJ,YV,A,n), hubs
end
