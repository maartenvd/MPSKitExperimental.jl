"""
    FiniteMPOHamiltonian(ham::FusedMPOHamiltonian)

The same operator as a regular MPSKit `FiniteMPOHamiltonian` on exactly the same virtual channels,
so that MPSKit's own `JordanMPOTensor` code paths can be compared against the fused ones.

Every block `(lmask, lblock, e, rblock, rmask)` contributes `lblock[a] * rblock[b] * e` to the entry
`(a, b)` of the site tensor; blocks whose operator is a multiple of the identity become identity scalars.
"""
function MPSKit.FiniteMPOHamiltonian(ham::FusedMPOHamiltonian)
    return FiniteMPOHamiltonian(map(_jordan_mpotensor, ham.data))
end

function _identity_coefficient(e)
    space(e, 1) == space(e, 4)' || return nothing
    id_e = TensorMap(MPSKit.similar_braidingtensor(e))
    λ = dot(id_e, e) / dot(id_e, id_e)
    return norm(e - λ * id_e) <= 1.0e-12 * max(norm(e), 1) ? λ : nothing
end

function _jordan_mpotensor(blk::FusedSparseBlock{E}) where {E}
    P = blk.pspace
    Vl = MPSKit.SumSpace(collect(blk.domspaces))
    Vr = MPSKit.SumSpace(collect(adjoint.(blk.imspaces)))
    W = MPSKit.jordanmpotensortype(spacetype(P), Vector{E})(undef, Vl ⊗ P ← P ⊗ Vr)

    tensors = Dict{Tuple{Int, Int}, Any}()
    scalars = Dict{Tuple{Int, Int}, E}()
    for (lmask, lblock, e, rblock, rmask) in blk.blocks
        λ = _identity_coefficient(e)
        for (a, la) in zip(findall(lmask), lblock), (b, rb) in zip(findall(rmask), rblock)
            c = la * rb
            iszero(c) && continue
            if isnothing(λ)
                tensors[(a, b)] = haskey(tensors, (a, b)) ? axpy!(c, e, tensors[(a, b)]) : c * e
            else
                scalars[(a, b)] = get(scalars, (a, b), zero(E)) + c * λ
            end
        end
    end
    # an entry with both kinds of contribution has to be stored as a single tensor
    for (k, s) in collect(scalars)
        haskey(tensors, k) || continue
        axpy!(s, TensorMap(MPSKit.similar_braidingtensor(tensors[k])), tensors[k])
        delete!(scalars, k)
    end

    for ((a, b), t) in tensors
        W[a, 1, 1, b] = t
    end
    for ((a, b), s) in scalars
        W[a, 1, 1, b] = s
    end
    return W
end
