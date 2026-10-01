#=
S matrix solver

w_AB ≈ (H-E)|AB⟩ and w_BA ≈ (H-E)|BA⟩ are the asymptotic two-particle states with (H-E) applied. The scattering state is

    |Ψ⟩ = |W⟩ + |AB⟩ + S|BA⟩

and we minimize F(W,S) = ‖(H-E)|W⟩ + |w_AB⟩ + S|w_BA⟩‖², sweeping over the columns of W.

Which gives:

    P_A (H-E)^2 |W> + P_A (H-E)|w_AB> + P_A (H-E) S |w_BA> = 0
    ⟨w_BA|(H-E)|W ⟩ + ⟨w_BA|w_AB⟩ + S ⟨w_BA|w_BA⟩ = 0

S is fixed given W, and so the entire thing is a single equation in W
=#

# AC_eff voor (H-E) W
function scatter_cross_proj(row::Int, col::Int, W::LeftGaugedMW, w::LeftGaugedMW, H, E::Number, le, re)
    env = environments(W, (H, w), le, re)
    return ac_proj(row, col, W, env) - E * projdown(row, col, w, W)
end

# given W, determine S and the magnitude of our cost function
function _scatter_S(W, venvs, w_AB, w_BA, E, λ = 0.0)
    # evaluate S and F with W fixed, around column 1
    g_AB = scatter_cross_proj(1, 1, W, w_AB, venvs.H, E, venvs.le, venvs.re)
    g_BA = scatter_cross_proj(1, 1, W, w_BA, venvs.H, E, venvs.le, venvs.re)
    x = W.AC[1, 1]
    n_BA = real(dot(w_BA, w_BA))
    c = dot(w_BA, w_AB)
    S = -(dot(g_BA, x) + c) / n_BA
    F = real(dot(x, variance_proj(1, 1, W, venvs, E)) + λ * norm(x)^2 + 2 * real(dot(x, g_AB + S * g_BA)) +
             dot(w_AB, w_AB) + 2 * real(S * conj(c)) + abs2(S) * n_BA)
    return S, F
end

"""
    scatter_lsq!(W, H, E, w_AB, w_BA; maxiter=10, tol=1e-8, verbosity=1, λ=0, linalg=CG(...))

Minimize ‖(H-E)|W⟩ + |w_AB⟩ + S|w_BA⟩‖² + λ‖W‖² over the window W and the number S. Returns (W, S, F).
(H-E) has near-null directions inside the window space (window states with energy ≈ E) along which W is not
determined; a small λ keeps W from drifting along them.
"""
function scatter_lsq!(W::LeftGaugedMW, H, E::Number, w_AB::LeftGaugedMW, w_BA::LeftGaugedMW;
                      maxiter = 10, tol = 1e-8, verbosity = 1, λ = 0.0,
                      linalg = CG(; tol = 1e-10, maxiter = 500),
                      venvs = variance_environments(W, H))
    size(W, 1) == 1 || throw(ArgumentError("only unit cell 1 for now"))
    dim(auxiliaryspace(W)) == dim(auxiliaryspace(w_BA)) && length(sectors(auxiliaryspace(W))) == 1 &&
        dim(auxiliaryspace(W), only(sectors(auxiliaryspace(W)))) == 1 ||
        throw(ArgumentError("S is a scalar: the util leg must be a single sector with multiplicity one"))

    n_BA = real(dot(w_BA, w_BA))
    c = dot(w_BA, w_AB)
    (S, F) = _scatter_S(W, venvs, w_AB, w_BA, E, λ)

    for iter in 1:maxiter
        S_prev = S
        F_prev = F
        for col in [1:size(W, 2); (size(W, 2) - 1):-1:2]
            g_AB = scatter_cross_proj(1, col, W, w_AB, H, E, venvs.le, venvs.re)
            g_BA = scatter_cross_proj(1, col, W, w_BA, H, E, venvs.le, venvs.re)

            # (A - g_BA g_BA'/n_BA) x = -g_AB + g_BA c/n_BA
            rhs = -g_AB + g_BA * (c / n_BA)
            (x, convhist) = linsolve(rhs, W.AC[1, col], linalg) do y
                W.AC[1, col] = y
                variance_proj(1, col, W, venvs, E) + λ * y - g_BA * (dot(g_BA, y) / n_BA)
            end
            convhist.converged == 0 && verbosity > 0 &&
                @warn "scatter_lsq: linsolve at col $col did not converge ($(convhist.normres))"
            W.AC[1, col] = x
            S = -(dot(g_BA, x) + c) / n_BA
            verbosity > 1 && @info "  col $col: $(convhist.numops) matvecs, residual $(convhist.normres), S = $S"
        end
        (S, F) = _scatter_S(W, venvs, w_AB, w_BA, E, λ)
        verbosity > 0 && @info "scatter_lsq iter $iter: S = $S, |S| = $(abs(S)), F = $F, ‖W‖ = $(norm(W.AC[1, 1]))"
        flush(stderr) # these runs are long; make progress visible when logging to a file
        abs(S - S_prev) < tol && abs(F - F_prev) < tol * max(1, abs(F)) && break
    end
    return W, S, F
end

"""
    scatter_galerkin!(W, H, E, w_AB, w_BA, BA_asym; maxiter=10, tol=1e-8, verbosity=1)

Galerkin variant: instead of minimizing the residual r = (H-E)|W⟩ + |w_AB⟩ + S|w_BA⟩, demand that it is
orthogonal to the variational directions themselves (the column of W, and |BA⟩ for S):

    P_A (H-E)|W⟩ + P_A |w_AB⟩ + S P_A |w_BA⟩ = 0
    ⟨BA|(H-E)|W⟩ + ⟨BA|w_AB⟩ + S⟨BA|w_BA⟩ = 0,    with ⟨BA|(H-E)|W⟩ ≈ ⟨w_BA|W⟩

Only one power of (H-E) appears, so no H² environments are needed and sweeps are cheaper. Because E sits in the
two-particle continuum the local problems are indefinite (GMRES), and the sweep can settle on spurious fixed points;
use it as a cheap starting point for scatter_lsq!. Returns (W, S).
"""
function scatter_galerkin!(W::LeftGaugedMW, H, E::Number, w_AB::LeftGaugedMW, w_BA::LeftGaugedMW, BA_asym;
                           maxiter = 10, tol = 1e-8, verbosity = 1,
                           linalg = GMRES(; tol = 1e-10, maxiter = 50, krylovdim = 100),
                           le = environments(W.left_gs, H), re = le)
    size(W, 1) == 1 || throw(ArgumentError("only unit cell 1 for now"))
    d = dim(auxiliaryspace(W)) # S is a number, dot sums over the util leg
    cAB = tr(partialdot(BA_asym, w_AB)) / d
    cBA = tr(partialdot(BA_asym, w_BA)) / d
    S = -cAB / cBA
    for iter in 1:maxiter
        S_prev = S
        for col in [1:size(W, 2); (size(W, 2) - 1):-1:2]
            y = (-projdown(1, col, w_AB, W), -cAB)
            (sol, convhist) = linsolve(y, (W.AC[1, col], S), linalg) do (x, t)
                W.AC[1, col] = x
                env = environments(W, (H, W), le, re)
                v1 = ac_proj(1, col, W, env) - E * x + t * projdown(1, col, w_BA, W)
                v2 = dot(w_BA, W) / d + t * cBA
                (v1, v2)
            end
            convhist.converged == 0 && verbosity > 0 &&
                @warn "scatter_galerkin: linsolve at col $col did not converge ($(convhist.normres))"
            W.AC[1, col] = sol[1]
            S = sol[2]
        end
        verbosity > 0 && @info "scatter_galerkin iter $iter: S = $S, |S| = $(abs(S))"
        flush(stderr)
        abs(S - S_prev) < tol && break
    end
    return W, S
end
