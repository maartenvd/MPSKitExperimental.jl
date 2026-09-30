# MPSKit removed the old ∂∂AC/∂∂C mechanism this package used to hijack, and replaced it
# with a "Jordan-block" decomposition of the MPO effective Hamiltonian:
# `MPSKit.AC_hamiltonian(site, below, operator::MPOHamiltonian, above, envs)` returns a
# `JordanMPO_AC_Hamiltonian` whose six pieces (D onsite, I not-started, E finished,
# C starting, B ending, A continuing) get summed on every call:
#   y = A(x); y += x*D; y += E*x; y += x*I; y += x*C; y += B*x
# Crucially, `prepare_operator!!` doesn't just prepare `A` — it also *folds* D into C or B,
# and I into C, and E into B (via identity-tensor absorption), whenever the target exists.
# For a typical bulk site this collapses 5 runtime terms down to just 2 (C and B). Skipping
# that folding and tightloop-compiling all 5 raw terms separately would mean doing strictly
# more contractions than MPSKit's own prepared path — not a fair comparison. So this file
# replicates the folding step exactly, then tightloop-compiles whatever terms remain.
# `A` (the "continuing" term) already delegates to MPSKit's own precomputed/prepared
# derivative operator, which is itself a fused-and-cached fast path — there is nothing left
# for tightloop to usefully accelerate there.
#
# Verdict (see examples/tighthack.ipynb): on par with MPSKit's own `AC_hamiltonian`, not faster.
# An earlier ~1.7x gap was not the keyword-argument calling convention: MPSKit's @plansor uses
# the non-planar @tensor path for Bosonic sectors (a single dense TensorOperations call for
# Trivial sectors), whereas this used @tightloop_planar throughout. With the kernel chosen the
# same way as @plansor, both run the same operations and only cache lookups are saved.

struct TightJordanAC{TA,TD,TE,TI,TC,TB,FD,FE,FI,FC,FB} <: MPSKit.DerivativeOperator
    A::TA
    D::TD; fD::FD
    E::TE; fE::FE
    I::TI; fI::FI
    C::TC; fC::FC
    B::TB; fB::FB
end

function (H::TightJordanAC)(x)
    y = ismissing(H.A) ? zerovector(x) : H.A(x)
    ismissing(H.D) || H.fD(x=x, D=H.D, y=y)
    ismissing(H.E) || H.fE(x=x, E=H.E, y=y)
    ismissing(H.I) || H.fI(x=x, I=H.I, y=y)
    ismissing(H.C) || H.fC(x=x, C=H.C, y=y)
    ismissing(H.B) || H.fB(x=x, B=H.B, y=y)
    return y
end

# mirrors MPSKit.JordanMPO_AC_Hamiltonian's own construction (algorithms/derivatives/hamiltonian_derivatives.jl),
# since the ∂∂AC-hijack pattern this package used to rely on can't call "the original AC_hamiltonian"
# from inside an active hijack without infinite recursion.
function tight_AC_hamiltonian(site::Int, below, operator::MPSKit.MPOHamiltonian, above, envs; prepare::Bool=true)
    @assert below === above "JordanMPO assumptions break"
    GL = MPSKit.leftenv(envs, site, below)
    GR = MPSKit.rightenv(envs, site, below)
    W = operator[site]

    D = MPSKit.nonzero_length(W.D) > 0 ? only(W.D) : missing

    I = size(W, 4) == 1 ? missing : removeunit(GR[1], 2)
    E = size(W, 1) == 1 ? missing : removeunit(GL[end], 2)

    C = if MPSKit.nonzero_length(W.C) > 0
        GR_2 = GR[2:(end - 1)]
        @plansor starting[-1 -2; -3 -4] := W.C[-1; -3 1] * GR_2[-4 1; -2]
        only(starting)
    else
        missing
    end

    B = if MPSKit.nonzero_length(W.B) > 0
        GL_2 = GL[2:(end - 1)]
        @plansor ending[-1 -2; -3 -4] := GL_2[-1 1; -3] * W.B[1 -2; -4]
        only(ending)
    else
        missing
    end

    # empty at the edges of a finite chain, where it cannot be prepared
    A = if MPSKit.nonzero_length(W.A) == 0
        missing
    else
        Araw = MPSKit.MPO_AC_Hamiltonian(GL[2:(end - 1)], W.A, GR[2:(end - 1)])
        prepare ? MPSKit.prepare_operator!!(Araw) : Araw
    end

    # mirror MPSKit's own prepare_operator!! folding, so we run exactly as many terms as it does
    if prepare
        if !ismissing(D)
            if !ismissing(C)
                Id = TensorKit.id(storagetype(C), space(C, 2))
                @plansor C[-1 -2; -3 -4] += D[-1; -3] * Id[-2; -4]
                D = missing
            elseif !ismissing(B)
                Id = TensorKit.id(storagetype(B), space(B, 1))
                @plansor B[-1 -2; -3 -4] += Id[-1; -3] * D[-2; -4]
                D = missing
            end
        end
        if !ismissing(I) && !ismissing(C)
            Id = TensorKit.id(storagetype(C), space(C, 1))
            @plansor C[-1 -2; -3 -4] += Id[-1; -3] * I[-4; -2]
            I = missing
        end
        if !ismissing(E) && !ismissing(B)
            Id = TensorKit.id(storagetype(B), space(B, 2))
            @plansor B[-1 -2; -3 -4] += E[-1; -3] * Id[-2; -4]
            E = missing
        end
    end

    x0 = below.AC[site]
    xtd = (typeof(x0), space(x0))

    # same choice as @plansor: non-planar contractions for Bosonic sectors, planar otherwise
    if BraidingStyle(sectortype(x0)) isa Bosonic
        fD = ismissing(D) ? missing : tight_D_apply_tensor(x=xtd, D=(typeof(D),space(D)), y=xtd)
        fE = ismissing(E) ? missing : tight_E_apply_tensor(x=xtd, E=(typeof(E),space(E)), y=xtd)
        fI = ismissing(I) ? missing : tight_I_apply_tensor(x=xtd, I=(typeof(I),space(I)), y=xtd)
        fC = ismissing(C) ? missing : tight_C_apply_tensor(x=xtd, C=(typeof(C),space(C)), y=xtd)
        fB = ismissing(B) ? missing : tight_B_apply_tensor(x=xtd, B=(typeof(B),space(B)), y=xtd)
    else
        fD = ismissing(D) ? missing : tight_D_apply_planar(x=xtd, D=(typeof(D),space(D)), y=xtd)
        fE = ismissing(E) ? missing : tight_E_apply_planar(x=xtd, E=(typeof(E),space(E)), y=xtd)
        fI = ismissing(I) ? missing : tight_I_apply_planar(x=xtd, I=(typeof(I),space(I)), y=xtd)
        fC = ismissing(C) ? missing : tight_C_apply_planar(x=xtd, C=(typeof(C),space(C)), y=xtd)
        fB = ismissing(B) ? missing : tight_B_apply_planar(x=xtd, B=(typeof(B),space(B)), y=xtd)
    end

    return TightJordanAC(A, D,fD, E,fE, I,fI, C,fC, B,fB)
end
