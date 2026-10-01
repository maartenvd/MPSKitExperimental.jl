#=
Everything needed to evaluate ⟨W|(H-E)²|W⟩ for a momentum window W (E measured relative to the groundstate),
and its derivative with respect to W.AC[row,col].

H is regularized (the groundstate energy density is subtracted) and squared. The environments of H_reg² are
regularized at every identity level, which drops the extensive groundstate contribution; what is left over is
the connected part up to a term 2*E_f*⟨W|H|W⟩, with E_f the remaining boundary energy of the regularized H.
Validated against a brute-force ring for AKLT and the transverse field ising model.
=#

struct VarianceEnvs{O1,O2,E1,E2}
    H::O1
    H2::O2 # (H - e_local)^2
    E_f::ComplexF64

    le::E1 # infinite groundstate environments of H
    re::E1
    le2::E2 # infinite groundstate environments of H2
    re2::E2
end

function variance_environments(state::LeftGaugedMW, H::InfiniteMPOHamiltonian, le = environments(state.left_gs, H))
    MPSKit.istopological(state) &&
        throw(ArgumentError("variance of domain wall excitations is not implemented"))
    gs = state.left_gs

    e_local = map(1:length(gs)) do i
        GL = leftenv(le, i, gs)
        GR = rightenv(le, i, gs)
        return MPSKit.contract_mpo_expval(gs.AC[i], GL, H[i][:, :, :, end], GR[end])
    end
    lattice = physicalspace(H)
    H_regularized = H - InfiniteMPOHamiltonian(
        lattice, i => e * id(storagetype(eltype(H)), lattice[i]) for (i, e) in enumerate(e_local)
    )

    rescaled_envs = environments(gs, H_regularized)
    GL = leftenv(rescaled_envs, 1, gs)
    GR = rightenv(rescaled_envs, 0, gs)
    E_f = @plansor GL[5 3; 1] * gs.C[0][1; 4] * conj(gs.C[0][5; 2]) * GR[4 3; 2]

    H2 = H_regularized^2
    le2 = environments(gs, H2)

    VarianceEnvs(H, H2, ComplexF64(E_f), le, le, le2, le2)
end

# (H-E)² acting on state.AC[row,col], with all other tensors fixed
function variance_proj(row::Int, col::Int, state::LeftGaugedMW, venvs::VarianceEnvs, E::Number)
    envs = environments(state, venvs.H, venvs.le, venvs.re)
    envs2 = environments(state, venvs.H2, venvs.le2, venvs.re2)

    y1 = ac_proj(row, col, state, envs)
    y2 = ac_proj(row, col, state, envs2)
    return y2 - 2 * (venvs.E_f + E) * y1 + E^2 * state.AC[row, col]
end

# the pieces ⟨W|W⟩, ⟨W|H|W⟩ and the connected ⟨W|H²|W⟩, evaluated around column col
function variance_moments(state::LeftGaugedMW, venvs::VarianceEnvs; col = 1)
    envs = environments(state, venvs.H, venvs.le, venvs.re)
    envs2 = environments(state, venvs.H2, venvs.le2, venvs.re2)

    n = sum(row -> norm(state.AC[row, col])^2, 1:size(state, 1))
    h1 = sum(row -> dot(state.AC[row, col], ac_proj(row, col, state, envs)), 1:size(state, 1))
    h2 = sum(row -> dot(state.AC[row, col], ac_proj(row, col, state, envs2)), 1:size(state, 1))
    return n, real(h1), real(h2 - 2 * venvs.E_f * h1)
end

function MPSKit.variance(state::LeftGaugedMW, H::InfiniteMPOHamiltonian, venvs::VarianceEnvs = variance_environments(state, H))
    (n, h1, h2) = variance_moments(state, venvs)
    return h2 / n - (h1 / n)^2
end
