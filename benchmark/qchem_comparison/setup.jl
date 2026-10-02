include("common.jl")

header()
(ERI, K, E0, NORB, NELEC, MS2) = parse_fcidump(joinpath(EXAMPLES, FCIDUMP))

t_opsum = @elapsed ham_opsum = opsum_su2_hamiltonian(E0, K, ERI)
t_fused = @elapsed (ham_fused, _) = fused_quantum_chemistry_hamiltonian(E0, K, ERI, Float64)
@printf("built OpSum MPO in %.1f s, fused MPO in %.1f s (both include compilation)\n", t_opsum, t_fused)

S = Irrep[U₁] ⊠ Irrep[SU₂] ⊠ FermionParity
psp = Vect[S]((0, 0, 0) => 1, (1, 1 // 2, 1) => 1, (2, 0, 0) => 1)
left = Vect[S]((-NELEC, MS2 // 2, mod(-NELEC, 2)) => 1)
virtual = Vect[S]((i, s, b) => 1 for i in -NELEC:0, s in 0:(1 // 2):(MS2 // 2 + 1), b in (0, 1))
st0 = FiniteMPS(rand, Float64, NORB, psp, virtual; left, right = oneunit(left))

hams = Dict{String, Any}(
    "opsum" => realify(ham_opsum), "fused" => ham_fused, "fused_jordan" => FiniteMPOHamiltonian(ham_fused)
)
energies = Dict(k => real(expectation_value(st0, H, fused_environments(st0, H))) for (k, H) in hams)
for (k, e) in energies
    @printf("⟨H⟩ on a random MPS, %-13s - fused: %.2e\n", k, e - energies["fused"])
end
mpo_summary(stdout, hams, NORB)

# the reference state only fixes the bond dimensions the benchmarks run at; fused_jordan is the fastest way to get it
sweeps = parse(Int, get(ENV, "SWEEPS", "3"))
alg = DMRG2(; trunc = truncrank(BOND), maxiter = sweeps, tol = 1.0e-14, verbosity = 2)
st, = find_groundstate(st0, hams["fused_jordan"], alg)
println("reference state bond dimensions: ", [dim(right_virtualspace(st, i)) for i in 1:NORB])

mkpath(dirname(DATAFILE))
serialize(DATAFILE, (; st, NORB, ham_opsum, ham_fused))
println("saved ", DATAFILE)
