using Revise,MPSKit, TensorKit, MPSKitExperimental, MPSKitModels, KrylovKit, LinearAlgebra, Plots

# create the AKLT hamiltonian
V_phys = SU2Space(1 => 1)
h_local = zeros(ComplexF64, V_phys ⊗ V_phys ← V_phys ⊗ V_phys)
block(h_local, SU2Irrep(2)) .= 1.0
th = @mpoham sum(h_local{i,i+1} for i in -Inf:Inf)

# find the groundstate
ts = InfiniteMPS(physicalspace(th),[Rep[SU₂](1//2=>1)]);
(ts,gs_env) = find_groundstate(ts,th,VUMPS(maxiter=200));

# first get the energy at k = pi. a window with length 5 is large enought to resolve this gap
len=5
windowchi = Rep[SU₂](1//2=>5,3//2=>5,5//2=>5); # bond dimension of the momentumwindow
window = LeftGaugedMW(rand,len,windowchi,ts,momentum = Float64(pi)+0im,utilspace= Rep[SU₂](1 => 1));    
aklt_gap = 0.0+0im
for top in 1:3
    for i in 1:len
        (vals,vecs, convhist) = eigsolve(window.AC[1,i],1,:SR) do x
            window.AC[1,i] = x
            envs = environments(window,th);
            MPSKitExperimental.ac_proj(1,i,window,envs)
        end

        aklt_gap = vals[1]
        window.AC[1,i] = vecs[1]
    end
end


# then do the same at k=0, for different window sizes
lens = collect(1:10)
energies = []
variances = []
for len in lens

    @show len
    window = LeftGaugedMW(rand,len,windowchi,ts,momentum = Float64(0)+0im,utilspace= Rep[SU₂](1 => 1));    
    en = 0.0+0im
    for top in 1:3
        for i in 1:len
            (vals,vecs, convhist) = eigsolve(window.AC[1,i],1,:SR) do x
                window.AC[1,i] = x
                envs = environments(window,th);
                MPSKitExperimental.ac_proj(1,i,window,envs)
            end

            en = vals[1]
            window.AC[1,i] = vecs[1]
        end
    end
    push!(energies,en)
    push!(variances, variance(window,th))
end

plot(variances,real.(energies).-real(2*aklt_gap),legend=false, seriestype=:scatter)
savefig("aklt_gap_convergence.png")