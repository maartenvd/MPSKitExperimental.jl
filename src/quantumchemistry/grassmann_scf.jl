#=
Orbital optimization for the quantum chemistry hamiltonian.

The hamiltonian is given by MO integrals (E0,K,V) in a reference basis (for example read from an FCIDUMP).
Rotating the orbitals by an orthogonal U gives the integrals (E0, UᵀKU, V×₁U×₂U×₃U×₄U), and the energy
    E(ψ,U) = E0 + K(U)⋅dK(ψ) + V(U)⋅dV(ψ)
is linear in the integrals, with dK/dV the one and two body reduced density matrices of ψ in the convention
of quantum_chemistry_hamiltonian (see qchem_rdms).

Two ways to minimize E(ψ,U):
- GrassmannSCF: co-optimize ψ (as a point on the grassmann manifold) and U (on SO(N)) in one conjugate gradient
- DMRGSCF: alternate between optimizing ψ with DMRG, and fully optimizing U at fixed dK/dV (cheap, no MPS work)
=#

# ---------------------------------------------------------------------------------------------
# reduced density matrices
# ---------------------------------------------------------------------------------------------

"""
    qchem_rdms(state) -> (dV,dK)

one and two body reduced density matrices of `state`, such that the energy of the hamiltonian
`quantum_chemistry_hamiltonian(E0,K,V)` is `E0 + sum(K.*dK) + sum(V.*dV)`.

They are the derivatives of the energy with respect to the integrals: the integrals only appear in the weights of
the channels of the hamiltonian, so this is `channel_gradient` of the symbolic qchem structure (one extra
environment sweep).
"""
const _qchem_gradient_structures = Dict{Tuple{Int,DataType},Any}()
const _qchem_gradient_structures_lock = ReentrantLock()

function qchem_rdms(state)
    N = length(state)
    T = real(scalartype(state))
    S = @lock _qchem_gradient_structures_lock get!(_qchem_gradient_structures,(N,T)) do
        (sym,ns,_) = qchem_structure(N,T)
        (pruned,kept) = prune_channels(sym,ns)
        nb = length(kept)
        GradientStructure(pruned,length.(kept);nparams = qchem_nparameters(N),
                          start = [b == nb ? 1 : findfirst(==(1),kept[b]) for b in 1:nb],
                          done = [b == 1 ? 1 : findfirst(==(ns[b]),kept[b]) for b in 1:nb])
    end
    (g,_) = channel_gradient(normalize(state),S)
    dK = reshape(g[2:1+N^2],N,N)
    dV = reshape(g[2+N^2:end],N,N,N,N)
    (dV,dK)
end

# ---------------------------------------------------------------------------------------------
# orbital rotations
# ---------------------------------------------------------------------------------------------

"""
MO integrals in a reference basis, the hamiltonian is E0 + ∑ K[i,j] c⁺ᵢcⱼ + ∑ V[i,j,k,l] c⁺ᵢc⁺ⱼcₖcₗ
"""
struct QChemIntegrals{T<:Real}
    E0::T
    K::Matrix{T}
    V::Array{T,4}
end
QChemIntegrals(E0,K,V) = QChemIntegrals(real(E0),Matrix(real(K)),Array(real(V)))

Base.length(ints::QChemIntegrals) = size(ints.K,1)

function rotate_integrals(ints::QChemIntegrals,U::AbstractMatrix)
    K = U'*ints.K*U
    @tensor V[p,q,r,s] := ints.V[a,b,c,d]*U[a,p]*U[b,q]*U[c,r]*U[d,s]
    QChemIntegrals(ints.E0,K,V)
end

rdm_energy(ints::QChemIntegrals,dV,dK) = ints.E0 + dot(ints.K,dK) + dot(ints.V,dV)

qchem_mpo(ints::QChemIntegrals) = quantum_chemistry_hamiltonian(ints.E0,ints.K,ints.V)

#=
Active spaces (as in the first version of this file, commit 2847980): orbitals 1:first(active)-1 are frozen and
doubly occupied, `active` is treated by the mps, last(active)+1:end are frozen and empty. The frozen orbitals are
still rotated, so the orbital optimization works in the full space with the rdms embedded in it.
=#

"""
    active_space(ints,active) -> QChemIntegrals

integrals of the active space, with the doubly occupied orbitals folded into E0 and K (mean field)
"""
function active_space(ints::QChemIntegrals,active::UnitRange{Int})
    (K,V) = (ints.K,ints.V)
    closed = 1:first(active)-1
    E = ints.E0
    for a in closed, b in closed
        if a == b
            E += 2*K[a,a] + 2*V[a,a,a,a]
        else
            E += 4*V[a,b,b,a] - 2*V[a,b,a,b]
        end
    end
    Kₐ = K[active,active]
    for a in closed
        Kₐ += 2*V[active,a,a,active] + 2*V[a,active,active,a] - V[a,active,a,active] - V[active,a,active,a]
    end
    QChemIntegrals(E,Kₐ,V[active,active,active,active])
end

"""
    embed_rdms(dV,dK,N,active) -> (dV,dK)

embed active space rdms in the full space of `N` orbitals, adding the doubly occupied orbitals 1:first(active)-1.
`rdm_energy(ints,embed_rdms(dV,dK,N,active)...) == rdm_energy(active_space(ints,active),dV,dK)`
"""
function embed_rdms(odV,odK,N,active::UnitRange{Int})
    dV = zeros(eltype(odV),N,N,N,N)
    dK = zeros(eltype(odK),N,N)
    dV[active,active,active,active] .= odV
    dK[active,active] .= odK
    for c in 1:first(active)-1
        dK[c,c] = 2
        dV[c,c,c,c] += 2
        for n in 1:first(active)-1
            n == c && continue
            dV[n,c,c,n] += 4
            dV[c,n,c,n] -= 2
        end
        dV[active,c,c,active] .+= 2*odK
        dV[c,active,active,c] .+= 2*odK
        dV[c,active,c,active] .-= odK
        dV[active,c,active,c] .-= odK
    end
    (dV,dK)
end

#=
E(W) = E0 + ∑ K[a,b] W[a,p] W[b,q] dK[p,q] + ∑ V[a,b,c,d] W[a,p] W[b,q] W[c,r] W[d,s] dV[p,q,r,s] is a polynomial in W,
with one factor of W per integral slot. Its euclidean gradient/hessian at W = 1 are sums over (pairs of) slots:
_slotgrad(T,P,i) is the contribution of slot i to ∂E/∂W, _slotrot(T,κ,j) applies κ to slot j of the integrals.
=#
_slotgrad(K::AbstractMatrix,D,i) = i == 1 ? K*transpose(D) : transpose(K)*D
_slotrot(K::AbstractMatrix,κ,j) = j == 1 ? transpose(κ)*K : K*κ
function _slotgrad(V::AbstractArray{<:Any,4},P,i)
    i == 1 && return @tensor G[a,p] := V[a,q,r,s]*P[p,q,r,s]
    i == 2 && return @tensor G[a,p] := V[q,a,r,s]*P[q,p,r,s]
    i == 3 && return @tensor G[a,p] := V[q,r,a,s]*P[q,r,p,s]
    return @tensor G[a,p] := V[q,r,s,a]*P[q,r,s,p]
end
function _slotrot(V::AbstractArray{<:Any,4},κ,j)
    j == 1 && return @tensor W[p,q,r,s] := V[a,q,r,s]*κ[a,p]
    j == 2 && return @tensor W[p,q,r,s] := V[p,a,r,s]*κ[a,q]
    j == 3 && return @tensor W[p,q,r,s] := V[p,q,a,s]*κ[a,r]
    return @tensor W[p,q,r,s] := V[p,q,r,a]*κ[a,s]
end
_euclidean_grad(T,P) = sum(i -> _slotgrad(T,P,i),1:ndims(T))
function _euclidean_hess(T,P,κ)
    sum(1:ndims(T)) do j
        Tκ = _slotrot(T,κ,j)
        sum(i -> _slotgrad(Tκ,P,i),filter(!=(j),1:ndims(T)))
    end
end
_euclidean_grad(ints::QChemIntegrals,dV,dK) = _euclidean_grad(ints.K,dK) + _euclidean_grad(ints.V,dV)

"""
    orbital_gradient(ints,dV,dK)

gradient of E(U exp(κ)) with respect to the antisymmetric κ, at κ = 0 (ints are the integrals rotated by U)
"""
function orbital_gradient(ints::QChemIntegrals,dV,dK)
    G = _euclidean_grad(ints,dV,dK)
    (G-G')/2
end

"""
    orbital_hessian(ints,dV,dK) -> κ ↦ Hκ

exact hessian of E(U exp(κ)) at κ = 0 and fixed rdms, as a matrix free linear map on antisymmetric matrices.
exp(κ) = 1 + κ + κ²/2 + ..., so on top of the euclidean hessian there is a term from ⟨G,κ²⟩/2.
Every application costs a handful of O(N⁵) integral contractions.
"""
function orbital_hessian(ints::QChemIntegrals,dV,dK)
    G = _euclidean_grad(ints,dV,dK)
    function (κ)
        H = _euclidean_hess(ints.K,dK,κ) + _euclidean_hess(ints.V,dV,κ) - (G*κ + κ*G)/2
        (H-H')/2
    end
end

# newton step, solved to the accuracy of the gradient (as in the old orbopt.jl). Away from a minimum the hessian
# can be indefinite, in which case we fall back to the plain gradient.
function orb_precondition(hessian,g)
    tol = max(norm(g)/10,eps(real(eltype(g))))
    (y,_) = linsolve(hessian,g,g,GMRES(;tol,krylovdim = 30,maxiter = 5,verbosity = 0))
    y = (y-y')/2
    return real(dot(g,y)) > 0 ? y : g
end

# U ↦ U exp(ακ), tangent vectors are antisymmetric matrices in the frame of U ("body frame").
# Parallel transport of the bi-invariant metric along that geodesic is conjugation with exp(ακ/2).
orb_retract(U,κ,α) = (U*exp(α*κ),κ)
function orb_transport(h,κ,α)
    ϵ = exp((α/2)*κ)
    ϵ'*h*ϵ
end

"""
    optimize_orbitals(ints,dV,dK; U=I, tol, maxiter, verbosity, precondition) -> (U,E,iterations)

minimize E0 + K(U)⋅dK + V(U)⋅dV over U, at fixed reduced density matrices. Every iteration only costs
an integral transformation.
"""
function optimize_orbitals(ints::QChemIntegrals,dV,dK;U = Matrix{eltype(ints.K)}(I,length(ints),length(ints)),
        tol = 1e-8, maxiter = 500, verbosity = 1, precondition = true)
    # the point carries its rotated integrals, so that the preconditioner can reuse them
    fg((U,rints)) = (rdm_energy(rints,dV,dK),orbital_gradient(rints,dV,dK))
    retract((U,_),κ,α) = let (U′,h) = orb_retract(U,κ,α)
        ((U′,rotate_integrals(ints,U′)),h)
    end
    prec((_,rints),g) = precondition ? orb_precondition(orbital_hessian(rints,dV,dK),g) : g
    ((U,_),E,_,_,history) = optimize(fg,(U,rotate_integrals(ints,U)),LBFGS(20;gradtol = tol,maxiter,verbosity);
        retract, inner = (x,a,b)->dot(a,b), transport! = (h,x,κ,α,x′)->orb_transport(h,κ,α),
        precondition = prec, scale! = (a,α)->rmul!(a,α), add! = (a,b,α)->axpy!(α,b,a), isometrictransport = true)
    (U,E,length(history)-1)
end

# ---------------------------------------------------------------------------------------------
# co-optimization of the mps and the orbitals
# ---------------------------------------------------------------------------------------------

"""
    GrassmannSCF(; tol, maxiter, verbosity)

Conjugate gradient in the product manifold (grassmann mps) × SO(norb): the mps and orbitals are optimized simultaneously.
The mps part is preconditioned with the inverse of the (regularized) density matrix, as in GradientGrassmann.

`find_groundstate(ψ,ints,alg;U,active)`: ψ covers the orbitals `active`, the ones before are doubly occupied,
the ones after are empty.
"""
struct GrassmannSCF <: MPSKit.Algorithm
    maxiter::Int
    tol::Float64
    verbosity::Int
    orb_precondition::Bool # newton-precondition the orbital part with the exact orbital hessian
end
GrassmannSCF(;tol = 1e-8,maxiter = 100,verbosity = 2,orb_precondition = true) = GrassmannSCF(maxiter,tol,verbosity,orb_precondition)

# a point on the manifold, with everything that is needed to evaluate the energy, its gradient and the preconditioner
struct SCFPoint{S,T,H,E}
    ψ::S
    U::Matrix{T}
    ints::QChemIntegrals{T} # rotated integrals
    ham::H
    envs::E
    dV::Array{T,4}
    dK::Matrix{T}
end

function SCFPoint(ψ,U,ref::QChemIntegrals,active)
    ints = rotate_integrals(ref,U)
    ham = qchem_mpo(active_space(ints,active))
    (dV,dK) = embed_rdms(qchem_rdms(ψ)...,length(ref),active)
    SCFPoint(ψ,U,ints,ham,disk_environments(ψ,ham),dV,dK)
end

function scf_fg(x::SCFPoint)
    (ψ,ham,envs) = (x.ψ,x.ham,x.envs)
    E = real(expectation_value(ψ,ham,envs))
    gψ = map(1:length(ψ)) do i
        AC′ = MPSKit.AC_hamiltonian(i,ψ,ham,ψ,envs)*ψ.AC[i]
        GrassmannMPS.rmul(Grassmann.project(AC′,ψ.AL[i]),ψ.C[i]')
    end
    (E,(gψ,orbital_gradient(x.ints,x.dV,x.dK)))
end

function MPSKit.find_groundstate(ψ::FiniteMPS,ref::QChemIntegrals,alg::GrassmannSCF;
        U = Matrix{eltype(ref.K)}(I,length(ref),length(ref)), active = 1:length(ref))
    length(ψ) == length(active) || throw(ArgumentError("the mps should cover the active space"))
    ψ = normalize!(copy(ψ))

    retract(x,g,α) = let (ψ′,hψ) = GrassmannMPS.retract(x.ψ,g[1],α), (U′,hU) = orb_retract(x.U,g[2],α)
        (SCFPoint(ψ′,U′,ref,active),(hψ,hU))
    end
    inner(x,a,b) = GrassmannMPS.inner(x.ψ,a[1],b[1]) + dot(a[2],b[2])
    transport!(h,x,g,α,x′) = (GrassmannMPS.transport!(h[1],x.ψ,g[1],α,x′.ψ),orb_transport(h[2],g[2],α))
    precondition(x,g) = (GrassmannMPS.precondition(x.ψ,g[1]),
        alg.orb_precondition ? orb_precondition(orbital_hessian(x.ints,x.dV,x.dK),g[2]) : g[2])
    scale!(g,α) = (GrassmannMPS.scale!(g[1],α),rmul!(g[2],α))
    add!(a,b,α) = (GrassmannMPS.add!(a[1],b[1],α),axpy!(α,b[2],a[2]))

    (x,E,_,_,_) = optimize(scf_fg,SCFPoint(ψ,U,ref,active),ConjugateGradient(;gradtol = alg.tol,maxiter = alg.maxiter,verbosity = alg.verbosity);
        retract, inner, transport!, precondition, scale!, add!, isometrictransport = true)

    (x.ψ,x.U,E)
end

# ---------------------------------------------------------------------------------------------
# alternating: optimize the mps, then fully optimize the orbitals at fixed rdms, repeat
# ---------------------------------------------------------------------------------------------

"""
    DMRGSCF(; mps_alg, tol, maxiter, orb_tol, verbosity)

`find_groundstate(ψ,ints,alg;U,active)`: ψ covers the orbitals `active`, the ones before are doubly occupied,
the ones after are empty.

Alternates between `find_groundstate(ψ,H(U),mps_alg)` and `optimize_orbitals` at fixed reduced density matrices,
until the energy changes less than `tol`.
"""
struct DMRGSCF{A} <: MPSKit.Algorithm
    mps_alg::A
    maxiter::Int
    tol::Float64
    orb_tol::Float64
    verbosity::Int
end
DMRGSCF(;mps_alg = DMRG2(;trscheme = truncrank(50),maxiter = 2,verbosity = 0),maxiter = 50,tol = 1e-8,orb_tol = 1e-6,verbosity = 1) =
    DMRGSCF(mps_alg,maxiter,tol,orb_tol,verbosity)

function MPSKit.find_groundstate(ψ::FiniteMPS,ref::QChemIntegrals,alg::DMRGSCF;
        U = Matrix{eltype(ref.K)}(I,length(ref),length(ref)), active = 1:length(ref))
    length(ψ) == length(active) || throw(ArgumentError("the mps should cover the active space"))
    E = Inf
    for it in 1:alg.maxiter
        ints = rotate_integrals(ref,U)
        ham = qchem_mpo(active_space(ints,active))
        (ψ,_) = find_groundstate(ψ,ham,alg.mps_alg,disk_environments(ψ,ham))
        E_mps = real(expectation_value(ψ,ham))

        (dV,dK) = embed_rdms(qchem_rdms(ψ)...,length(ref),active)
        (U′,E′) = optimize_orbitals(ints,dV,dK;tol = alg.orb_tol,verbosity = 0)
        U = U*U′

        alg.verbosity > 0 && @info "DMRGSCF $it: E(mps) = $E_mps, E(orbitals) = $E′, |∇U| = $(norm(orbital_gradient(ints,dV,dK)))"
        abs(E-E′) < alg.tol && (E = E′; break)
        E = E′
    end
    (ψ,U,E)
end
