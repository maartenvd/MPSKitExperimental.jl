# Quantum-chemistry Hamiltonian with OpSum.jl, SU(2)-adapted: one site per spatial orbital,
# graded by U₁ ⊠ SU₂ ⊠ FermionParity (the same site space as qchem_n2.ipynb).
#
# Usage (e.g. from a notebook in examples/, with the output of MPSKitExperimental.parse_fcidump):
#
#   include("opsum_qchem_su2.jl"); using .OpSumQChemSU2
#   H = opsum_su2_hamiltonian(E0, K, ERI)          # FiniteMPOHamiltonian on NORB sites, E0 included
#   # physical space: OpSumQChemSU2.Vh == U₁⊠SU₂⊠fℤ₂ (0,0,0) ⊕ (1,½,1) ⊕ (2,0,0), as in qchem_n2.ipynb
#
# Runner and checks: opsum/opsum_qchem_su2.jl (DMRG on an FCIDUMP), opsum/su2_check.jl (spectra vs
# the spin-orbital construction, which is itself checked against Jordan–Wigner ED).
#
#   H = Σ_{pq} h_pq Σ_σ a†_pσ a_qσ + ½ Σ_{pqrs} (pq|rs) Σ_{στ} a†_pσ a†_rτ a_sτ a_qσ
#
# Terms are built with `couple` from single alphabet letters, nesting with explicit intermediate
# spins; nothing is projected. The coefficients come from SU(2) algebra done here:
#
#  1. An operator string is sorted by site (fermionic sign from the permutation), so it becomes a
#     site-ordered product of on-site operators M_k, each a product of a†_σ / a_σ on one orbital.
#  2. For every spin assignment, each M_k is expanded in the components L^m of the alphabet letters
#     (a 16-dimensional basis of End(V); L^m is the letter's reduced element times TensorKit's
#     Clebsch–Gordan array, so this is Wigner–Eckart in TensorKit's own conventions).
#  3. Summing over spins gives, per tuple of letters, an invariant tensor Y in ⊗ₖ C_k (the letters'
#     charge spaces). Its overlap with each caterpillar fusion tree X_T (again TensorKit's CG arrays)
#     is the coefficient of `couple(...; to = inner channels of T)`. A residual check asserts that Y
#     lies in the span of the trees.
#
# Steps 2-3 depend only on the *pattern* of a term (which of p,q,r,s coincide and their site order),
# so they are cached: a few dozen small CG contractions in total, independent of the orbital count.
#
# Sign fix: OpSum materialises a couple-built term with an extra (-1)^{p(c)·p(in)} per letter
# (charge parity times input-sector parity); see sign_check.jl / triangle_identity.jl. For the
# spin-orbital encoding that is "c → -c". Here c† has a letter with odd input (|↑↓⟩ ← |σ⟩), so the
# correction is applied per letter, not per operator.
module OpSumQChemSU2

export opsum_su2_hamiltonian, qchem_opsum_su2, from_parse_fcidump

using LinearAlgebra
using TensorKit
using OpSum: OpSum, couple, opsum, FiniteChain, jordan_mpo_tensors, instantiate, IrrepOperator,
    SiteOperator
using BlockTensorKit: nonzero_pairs
using MPSKit: MPSKit, FiniteMPOHamiltonian

const S2 = U1Irrep ⊠ SU2Irrep ⊠ FermionParity
const Vh = Vect[S2]((0, 0, 0) => 1, (1, 1 // 2, 1) => 1, (2, 0, 0) => 1)
su2(s::S2) = s.sectors[2]
oddp(s::S2) = s.sectors[3].isodd

# On-site m-basis: 1 = |0⟩, 2 = |σ=1⟩, 3 = |σ=2⟩, 4 = |↑↓⟩ := a†_1 a†_2 |0⟩, where σ = 1, 2 is
# TensorKit's index order within the spin-½ multiplet.
function stateindex(s::S2, m::Int)
    n = s.sectors[1].charge
    return n == 0 ? 1 : n == 1 ? 1 + m : 4
end
unitmat(i, j) = (M = zeros(4, 4); M[i, j] = 1; M)
const AD = (unitmat(2, 1) + unitmat(4, 3), unitmat(3, 1) - unitmat(4, 2))   # a†_σ
const AN = map(transpose, AD)                                                 # a_σ
for σ in 1:2, τ in 1:2      # on-site CAR
    @assert AD[σ] * AN[τ] + AN[τ] * AD[σ] ≈ (σ == τ) * Matrix{Float64}(LinearAlgebra.I, 4, 4)
    @assert AD[σ] * AD[τ] + AD[τ] * AD[σ] ≈ zeros(4, 4)
end

# Alphabet letters: components L^m as 4×4 matrices, and the per-letter sign fix.
struct Letter
    op::IrrepOperator{S2}
    comps::Array{Float64, 3}     # (out, in, m)
    sign::Int                    # (-1)^{p(c)·p(in)}
end
function Letter(op::IrrepOperator{S2})
    t = instantiate(op, Vh)                          # Vh ← Vh ⊗ Vect[c]
    dc = dim(su2(op.c))
    comps = zeros(4, 4, dc)
    sign = 1
    for (fo, fi) in fusiontrees(t)
        v = real(only(t[fo, fi]))
        iszero(v) && continue
        sin, c = fi.uncoupled
        sout = fi.coupled
        CG = TensorKit.fusiontensor(su2(sin), su2(c), su2(sout))
        for mi in 1:dim(su2(sin)), mc in 1:dc, mo in 1:dim(su2(sout))
            comps[stateindex(sout, mo), stateindex(sin, mi), mc] += v * CG[mi, mc, mo, 1]
        end
        sign = oddp(c) && oddp(sin) ? -1 : 1
    end
    return Letter(op, comps, sign)
end
const LETTERS = [Letter(op) for op in instances(IrrepOperator, Vh)]
const BASIS = [(ℓ, m) for ℓ in eachindex(LETTERS) for m in axes(LETTERS[ℓ].comps, 3)]
const BINV = let B = reduce(hcat, [vec(LETTERS[ℓ].comps[:, :, m]) for (ℓ, m) in BASIS])
    @assert size(B) == (16, 16) && rank(B) == 16
    inv(B)
end
function decompose(M)
    x = BINV * vec(M)
    return [(BASIS[i]..., x[i]) for i in eachindex(x) if abs(x[i]) > 1.0e-12]
end

# Caterpillar fusion tree (product sectors) → its SU(2) CG tensor, legs (m_1, …, m_K).
function treetensor(f)
    K = length(f.uncoupled)
    g = FusionTree{SU2Irrep}(map(su2, f.uncoupled), su2(f.coupled), ntuple(_ -> false, K),
                             map(su2, f.innerlines))
    return reshape(convert(Array, g), map(c -> dim(su2(c)), f.uncoupled))
end

# Pattern: per site, the operators acting there in order, as (:cd | :c, spin label).
# Returns [(letters, innerlines, coeff)] with the sign fix already folded into coeff.
const PATTERNS = Dict{Vector{Vector{Tuple{Symbol, Int}}}, Vector{Tuple{Vector{Int}, Vector{S2}, Float64}}}()
function pattern_terms(groups)
    return get!(PATTERNS, groups) do
        K = length(groups)
        nlabels = maximum(l for g in groups for (_, l) in g)
        Y = Dict{Vector{Int}, Array{Float64}}()
        for spins in Iterators.product(ntuple(_ -> 1:2, nlabels)...)
            parts = map(groups) do g
                decompose(prod((t === :cd ? AD : AN)[spins[l]] for (t, l) in g))
            end
            for combo in Iterators.product(parts...)
                ℓs = [x[1] for x in combo]
                arr = get!(Y, ℓs) do
                    zeros(Tuple(size(LETTERS[ℓ].comps, 3) for ℓ in ℓs))
                end
                arr[(x[2] for x in combo)...] += prod(x[3] for x in combo)
            end
        end
        out = Tuple{Vector{Int}, Vector{S2}, Float64}[]
        for (ℓs, y) in Y
            norm(y) < 1.0e-12 && continue
            cs = Tuple(LETTERS[ℓ].op.c for ℓ in ℓs)
            sum(c -> c.sectors[1].charge, cs) == 0 || continue
            fit = zero(y)
            for f in fusiontrees(cs, one(S2))
                X = treetensor(f)
                coeff = dot(X, y) / dot(X, X)
                abs(coeff) < 1.0e-12 && continue
                fit .+= coeff .* X
                push!(out, (ℓs, collect(S2, f.innerlines), coeff * prod(LETTERS[ℓ].sign for ℓ in ℓs)))
            end
            norm(fit - y) < 1.0e-10 || error("pattern $groups, letters $ℓs: not SU(2) invariant")
        end
        out
    end
end

# `string` is [(site, :cd | :c, spin label)] as written left to right.
function add_string!(acc, coeff, string)
    perm = sortperm(string; by = first)              # stable
    ninv = count(perm[i] > perm[j] for i in eachindex(perm) for j in (i + 1):lastindex(perm))
    s = string[perm]
    sites = unique(first.(s))
    groups = [[(t, l) for (st, t, l) in s if st == site] for site in sites]
    c = isodd(ninv) ? -coeff : coeff
    for (ℓs, inner, y) in pattern_terms(groups)
        key = (sites, ℓs, inner)
        acc[key] = get(acc, key, 0.0) + c * y
    end
    return acc
end

function build_term(sites, ℓs, inner, coeff)
    ops = [SiteOperator(LETTERS[ℓ].op)[st] for (ℓ, st) in zip(ℓs, sites)]
    K = length(ops)
    K == 1 && return coeff * ops[1]
    acc = ops[1]
    for k in 2:(K - 1)
        acc = couple(acc, ops[k]; to = inner[k - 1])
    end
    return coeff * couple(acc, ops[K])
end

function qchem_opsum_su2(h, eri; tol = 1.0e-14)
    N = size(h, 1)
    acc = Dict{Tuple{Vector{Int}, Vector{Int}, Vector{S2}}, Float64}()
    for p in 1:N, q in 1:N
        abs(h[p, q]) > tol && add_string!(acc, h[p, q], [(p, :cd, 1), (q, :c, 1)])
    end
    for p in 1:N, q in 1:N, r in 1:N, s in 1:N
        v = eri[p, q, r, s] / 2
        abs(v) > tol && add_string!(acc, v, [(p, :cd, 1), (r, :cd, 2), (s, :c, 2), (q, :c, 1)])
    end
    filter!(kv -> abs(kv[2]) > tol, acc)
    return opsum(build_term(k..., c) for (k, c) in acc), length(acc)
end

# `parse_fcidump` stores K = h and ERI[q, s, r, p] = (pq|rs) / 2 (checked against a plain
# chemist-notation reader on N2.STO3G.FCIDUMP).
function from_parse_fcidump(K, ERI)
    N = size(K, 1)
    h = real.(Matrix(K))
    eri = [2 * real(ERI[q, s, r, p]) for p in 1:N, q in 1:N, r in 1:N, s in 1:N]
    return h, eri
end

# The identity on one site, as a combination of charge-0 letters (to carry the constant E0).
function identity_op()
    out = zero(SiteOperator{S2})
    for (ℓ, m, x) in decompose(Matrix{Float64}(LinearAlgebra.I, 4, 4))
        out += x * SiteOperator(LETTERS[ℓ].op)
    end
    return out
end

# OpSum's Jordan-form tensors → MPSKit JordanMPOTensors (from OpSum's examples/mpskit.jl: the
# JordanMPOTensor constructor rejects boundary tensors, so go through undef + setindex!).
function to_jordan(W)
    O = MPSKit.jordanmpotensortype(spacetype(W), storagetype(W))(undef, space(W))
    for (ix, v) in nonzero_pairs(W)
        O[ix] = v
    end
    return O
end

"""
    opsum_su2_hamiltonian(E0, K, ERI; tol = 1e-14) -> FiniteMPOHamiltonian

Same arguments as `fused_quantum_chemistry_hamiltonian` (the output of `parse_fcidump`), E0 included.
"""
function opsum_su2_hamiltonian(E0, K, ERI; tol = 1.0e-14)
    h, eri = from_parse_fcidump(K, ERI)
    H, _ = qchem_opsum_su2(h, eri; tol)
    iszero(E0) || (H = H + real(E0) * identity_op()[1])
    Ws = jordan_mpo_tensors(H, FiniteChain(Vh, size(h, 1)))
    return FiniteMPOHamiltonian(map(to_jordan, Ws))
end

end
