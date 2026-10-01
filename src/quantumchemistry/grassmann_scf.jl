#=
Orbital optimization for the quantum chemistry hamiltonian.

The hamiltonian is given by MO integrals (E0,K,V) in a reference basis (for example read from an FCIDUMP).
Rotating the orbitals by an orthogonal U gives the integrals (E0, UᵀKU, V×₁U×₂U×₃U×₄U), and the energy
    E(ψ,U) = E0 + K(U)⋅dK(ψ) + V(U)⋅dV(ψ)
is linear in the integrals, with dK/dV the one and two body reduced density matrices of ψ in the convention
of fused_quantum_chemistry_hamiltonian (see quantum_chemistry_dV_dK).

Two ways to minimize E(ψ,U):
- GrassmannSCF: co-optimize ψ (as a point on the grassmann manifold) and U (on SO(N)) in one conjugate gradient
- DMRGSCF: alternate between optimizing ψ with DMRG, and fully optimizing U at fixed dK/dV (cheap, no MPS work)
=#

# ---------------------------------------------------------------------------------------------
# reduced density matrices
# ---------------------------------------------------------------------------------------------

# this function calculates the one and two body reduced density matrices. Because I was lazy, it is slow and makes use of the mpo-representation code
# conceptually this should be decoupled, and it would avoid subtle bugs that are now very much possible.
function quantum_chemistry_dV_dK(state,qchemham,indmaps,envs=disk_environments(state,qchemham))
    (indmap_1Ls, indmap_1Rs, indmap_2Ls, indmap_2Rs) = indmaps
    
    basis_size = length(state);

    half_basis_size = Int(ceil((basis_size+1)/2));
    
    Elt = scalartype(state.AL[1])

    dK = fill(zero(Elt),basis_size,basis_size);
    dV = fill(zero(Elt),basis_size,basis_size,basis_size,basis_size);
    psp = Vect[(Irrep[U₁]⊠Irrep[SU₂] ⊠ FermionParity)]((0,0,0)=>1, (1,1//2,1)=>1, (2,0,0)=>1);

    ap = ones(Elt,psp*Vect[(Irrep[U₁]⊠Irrep[SU₂] ⊠ FermionParity)]((-1,1//2,1)=>1),psp);
    block(ap,(Irrep[U₁](0)⊠Irrep[SU₂](0)⊠FermionParity(0))) .*= -sqrt(2);
    block(ap,(Irrep[U₁](1)⊠Irrep[SU₂](1//2)⊠FermionParity(1))) .*= 1;


    bm = ones(Elt,psp,Vect[(Irrep[U₁]⊠Irrep[SU₂]⊠FermionParity)]((-1,1//2,1)=>1)*psp);
    block(bm,(Irrep[U₁](0)⊠Irrep[SU₂](0)⊠FermionParity(0))) .*= sqrt(2);
    block(bm,(Irrep[U₁](1)⊠Irrep[SU₂](1//2)⊠FermionParity(1))) .*= -1;

    # this transposition is easier to reason about in a planar way
    am = transpose(ap',((2,1),(3,)));
    bp = transpose(bm',((1,),(3,2)));
    ap = transpose(ap,((3,1),(2,)));
    bm = transpose(bm,((2,),(3,1)));
    
    flipcor = isometry(flip(space(bm,1)),space(bm,1));
    bm = flipcor*bm;
    ap = ap*flipcor';

    @plansor b_derp[-1 -2;-3] := bp[1;2 -2]*τ[-3 -1;2 1]
    @plansor b_derp[-1 -2;-3] := bm[1;2 -2]*τ[-3 -1;2 1]

    h_pm = ones(Elt,psp,psp);
    block(h_pm,(Irrep[U₁](0)⊠Irrep[SU₂](0)⊠ FermionParity(0))) .=0;
    block(h_pm,(Irrep[U₁](1)⊠Irrep[SU₂](1//2)⊠ FermionParity(1))) .=1;
    block(h_pm,(Irrep[U₁](2)⊠Irrep[SU₂](0)⊠ FermionParity(0))) .=2;

    @plansor o_derp[-1 -2;-3 -4] := am[-1 1;-3]*ap[1 -2;-4]
    h_pm_derp = transpose(h_pm,((2,1),()));
    Lmap_apam_to_pm = find_right_map(o_derp,h_pm_derp)

    @plansor o_derp[-1 -2;-3 -4] := bm[-1;-3 1]*bp[-2;1 -4]
    h_pm_derp2 = transpose(h_pm,((),(2,1)));
    Rmap_bpbm_to_pm = find_left_map(o_derp,h_pm_derp2)

    h_ppmm = h_pm*h_pm-h_pm;
    
    ai = isomorphism(storagetype(ap),psp,psp);

    ut = ones(Elt,oneunit(psp));
    @plansor ut_ap[-1 -2; -3 -4] := ut[-1]*ap[-3 -2;-4];
    @plansor ut_am[-1 -2; -3 -4] := ut[-1]*am[-3 -2;-4];
    @plansor bp_ut[-1 -2; -3 -4] := bp[-1;-3 -2]*conj(ut[-4]);
    @plansor bm_ut[-1 -2; -3 -4] := bm[-1;-3 -2]*conj(ut[-4]);
    pp_f = isometry(fuse(_lastspace(ap)'*_lastspace(ap)'),_lastspace(ap)'*_lastspace(ap)');
    mm_f = isometry(fuse(_lastspace(am)'*_lastspace(am)'),_lastspace(am)'*_lastspace(am)');
    mp_f = isometry(fuse(_lastspace(am)'*_lastspace(ap)'),_lastspace(am)'*_lastspace(ap)');
    pm_f = isometry(fuse(_lastspace(ap)'*_lastspace(am)'),_lastspace(ap)'*_lastspace(am)');
    pp_f_1 = isometry(space(pp_f,1),Vect[(Irrep[U₁] ⊠ Irrep[SU₂] ⊠ FermionParity)]((2, 0, 0)=>1))'*pp_f;
    pp_f_2 = isometry(space(pp_f,1),Vect[(Irrep[U₁] ⊠ Irrep[SU₂] ⊠ FermionParity)]((2, 1, 0)=>1))'*pp_f;
    mm_f_1 = isometry(space(mm_f,1),Vect[(Irrep[U₁] ⊠ Irrep[SU₂] ⊠ FermionParity)]((-2, 0, 0)=>1))'*mm_f;
    mm_f_2 = isometry(space(mm_f,1),Vect[(Irrep[U₁] ⊠ Irrep[SU₂] ⊠ FermionParity)]((-2, 1, 0)=>1))'*mm_f;
    pm_f_1 = isometry(space(pm_f,1),Vect[(Irrep[U₁] ⊠ Irrep[SU₂] ⊠ FermionParity)]((0, 0, 0)=>1))'*pm_f;
    pm_f_2 = isometry(space(pm_f,1),Vect[(Irrep[U₁] ⊠ Irrep[SU₂] ⊠ FermionParity)]((0, 1, 0)=>1))'*pm_f;
    mp_f_1 = isometry(space(mp_f,1),Vect[(Irrep[U₁] ⊠ Irrep[SU₂] ⊠ FermionParity)]((0, 0, 0)=>1))'*mp_f;
    mp_f_2 = isometry(space(mp_f,1),Vect[(Irrep[U₁] ⊠ Irrep[SU₂] ⊠ FermionParity)]((0, 1, 0)=>1))'*mp_f;
    pp_f_1 = pp_f*pp_f_1'*pp_f_1;
    pp_f_2 = pp_f*pp_f_2'*pp_f_2;
    mm_f_1 = mm_f*mm_f_1'*mm_f_1;
    mm_f_2 = mm_f*mm_f_2'*mm_f_2;
    pm_f_1 = pm_f*pm_f_1'*pm_f_1;
    pm_f_2 = pm_f*pm_f_2'*pm_f_2;
    mp_f_1 = mp_f*mp_f_1'*mp_f_1;
    mp_f_2 = mp_f*mp_f_2'*mp_f_2;
    @plansor ut_apap[-1 -2; -3 -4] := ut[-1]*ap[-3 1;3]*ap[1 -2;4]*conj(pp_f[-4;3 4]);
    @plansor ut_amam[-1 -2; -3 -4] := ut[-1]*am[-3 1;3]*am[1 -2;4]*conj(mm_f[-4;3 4]);
    @plansor ut_amap[-1 -2; -3 -4] := ut[-1]*am[-3 1;3]*ap[1 -2;4]*conj(mp_f[-4;3 4]);
    @plansor ut_apam[-1 -2; -3 -4] := ut[-1]*ap[-3 1;3]*am[1 -2;4]*conj(pm_f[-4;3 4])
    @plansor bpbp_ut[-1 -2; -3 -4] := mm_f[-1;1 2]*bp[1;-3 3]*bp[2;3 -2]*conj(ut[-4]);
    @plansor bmbm_ut[-1 -2; -3 -4] := pp_f[-1;1 2]*bm[1;-3 3]*bm[2;3 -2]*conj(ut[-4]);
    @plansor bmbp_ut[-1 -2; -3 -4] := pm_f[-1;1 2]*bm[1;-3 3]*bp[2;3 -2]*conj(ut[-4]);
    @plansor bpbm_ut[-1 -2; -3 -4] := mp_f[-1;1 2]*bp[1;-3 3]*bm[2;3 -2]*conj(ut[-4]);
    iso_pp = isomorphism(_lastspace(ap)',_lastspace(ap)');
    iso_mm = isomorphism(_lastspace(am)',_lastspace(am)');
    @plansor p_ai_p[-1 -2; -3 -4] := iso_pp[-1;1]*τ[1 2;-3 -4]*ai[-2;2]
    @plansor m_ai_m[-1 -2; -3 -4] := iso_mm[-1;1]*τ[1 2;-3 -4]*ai[-2;2]
    @plansor p_pm_p[-1 -2; -3 -4] := iso_pp[-1;1]*τ[1 2;-3 -4]*h_pm[-2;2]
    @plansor m_pm_m[-1 -2; -3 -4] := iso_mm[-1;1]*τ[1 2;-3 -4]*h_pm[-2;2]
    iso_pppp = pp_f*pp_f';
    iso_pmpm = pm_f*pm_f';
    iso_mmmm = mm_f*mm_f';
    @plansor pp_ai_pp[-1 -2; -3 -4] := iso_pppp[-1;1]*τ[1 2;-3 -4]*ai[-2;2]
    @plansor pm_ai_pm[-1 -2; -3 -4] := iso_pmpm[-1;1]*τ[1 2;-3 -4]*ai[-2;2]
    @plansor mm_ai_mm[-1 -2; -3 -4] := iso_mmmm[-1;1]*τ[1 2;-3 -4]*ai[-2;2]
    @plansor p_ap[-1 -2; -3 -4] := iso_pp[-1;1]*τ[1 2;-3 3]*ap[2 -2;4]*conj(pp_f[-4;3 4]);
    @plansor m_ap[-1 -2; -3 -4] := iso_mm[-1;1]*τ[1 2;-3 3]*ap[2 -2;4]*conj(mp_f[-4;3 4]);
    @plansor p_am[-1 -2; -3 -4] := iso_pp[-1;1]*τ[1 2;-3 3]*am[2 -2;4]*conj(pm_f[-4;3 4]);
    @plansor m_am[-1 -2; -3 -4] := iso_mm[-1;1]*τ[1 2;-3 3]*am[2 -2;4]*conj(mm_f[-4;3 4]);
    @plansor bp_p[-1 -2; -3 -4] := bp[2;-3 3]*iso_mm[1;-4]*τ[4 -2;3 1]*mm_f[-1;2 4]
    @plansor bm_p[-1 -2; -3 -4] := bm[2;-3 3]*iso_mm[1;-4]*τ[4 -2;3 1]*pm_f[-1;2 4]
    @plansor bm_m[-1 -2; -3 -4] := bm[2;-3 3]*iso_pp[1;-4]*τ[4 -2;3 1]*pp_f[-1;2 4]
    @plansor bp_m[-1 -2; -3 -4] := bp[2;-3 3]*iso_pp[1;-4]*τ[4 -2;3 1]*mp_f[-1;2 4]
    @plansor ppLm[-1 -2; -3 -4] := bp[-1;1 -2]*h_pm[1;-3]*conj(ut[-4])
    @plansor Lpmm[-1 -2; -3 -4] := bm[-1;-3 1]*h_pm[-2;1]*conj(ut[-4])
    @plansor ppRm[-1 -2; -3 -4] := ut[-1]*ap[1 -2;-4]*h_pm[1;-3]
    @plansor Rpmm[-1 -2; -3 -4] := ut[-1]*h_pm[-2;1]*am[-3 1;-4]
    @plansor LRmm[-1 -2; -3 -4] := am[1 -2;-4]*bm[-1;-3 1]
    @plansor ppLR[-1 -2; -3 -4] := ap[1 -2;-4]*bp[-1;-3 1]
    @plansor LpRm[-1 -2; -3 -4] := ap[1 -2;-4]*bm[-1;-3 1]
    @plansor RpLm[-1 -2; -3 -4] := bp[-1;1 -2]*am[-3 1;-4]
    @plansor _pm_left[-1 -2; -3 -4] := (mp_f*Lmap_apam_to_pm)[-1]*h_pm[-2;-3]*conj(ut[-4])
    @plansor _pm_right[-1 -2; -3 -4] := ut[-1]*h_pm[-2;-3]*(transpose(Rmap_bpbm_to_pm*pm_f',((1,),())))[-4]
    @plansor LRLm_1[-1 -2; -3 -4] := (mp_f_1)[-1;1 2]*bm[2;3 -2]*τ[1 3;-3 -4]
    @plansor LpLR_1[-1 -2; -3 -4] := (mp_f_1)[-1;1 2]*bp[1;-3 3]*τ[3 2;-4 -2]
    @plansor RpLL[-1 -2; -3 -4] := mm_f[-1;1 2]*bp[2;3 -2]*τ[1 3;-3 -4]
    @plansor jimm[-1 -2; -3 -4] := iso_pp[-1;1]*ap[-3 2;3]*τ[2 1;4 -2]*conj(pp_f[-4;3 4])
    @plansor ppji[-1 -2; -3 -4] := iso_mm[-1;1]*am[-3 2;3]*τ[2 1;4 -2]*conj(mm_f[-4;3 4])
    @plansor jpim_1[-1 -2; -3 -4] := iso_mm[-1;1]*ap[-3 2;3]*τ[2 1;4 -2]*conj((pm_f_1)[-4;3 4])
    @plansor ipjm_1[-1 -2; -3 -4] := iso_pp[-1;1]*τ[-3 1;2 3]*am[3 -2;4]*conj((pm_f_1)[-4;2 4])
    @plansor jimR_1[-1 -2; -3 -4] := pp_f_1[-1;1 2]*τ[3 2;-4 -2]*bm[1;-3 3]
    @plansor jkil_2[-1 -2; -3 -4] := mp_f_2[-1;1 2]*ai[3;-3]*τ[3 1;4 5]*τ[5 2;6 -2]*conj(mp_f_2[-4;4 6])
    @plansor jikl_1[-1 -2; -3 -4] := pp_f_1[-1;1 2]*ai[3;-3]*τ[3 1;4 5]*τ[5 2;6 -2]*conj(pp_f_1[-4;4 6])
    @plansor lkij_1[-1 -2; -3 -4] := mm_f_1[-1;1 2]*ai[3;-3]*τ[3 1;4 5]*τ[5 2;6 -2]*conj(mm_f_1[-4;4 6])

    acc_lock = ReentrantLock()
    # the spawned tasks below accumulate into shared dV/dK
    for loc in 1:basis_size
        indmap_1L = indmap_1Ls[loc]
        indmap_2L = indmap_2Ls[loc]
        indmap_1R = indmap_1Rs[loc+1]
        indmap_2R = indmap_2Rs[loc+1]


        ac = state.AC[loc];
        ac_flipped = transpose(ac',((1,3),(2,)));

        le = leftenv(envs,loc,state);
        re = rightenv(envs,loc,state);
        l_end = length(le);
        r_end = length(re);

        # l/r environments are 3-leg tensors [-1 -2;-3]; the transposes are planar (cyclic),
        # so these contractions are equivalent to the @plansor ones in fused_env.jl
        lo(leftind::Int,e) = leftind == 0 ? 0 : lo(le[leftind],e)
        function lo(l::AbstractTensorMap,e)
            lAb = ac_flipped*transpose(l,((1,),(3,2)))
            lAbe = transpose(lAb,((3,1),(4,2)))*e
            return transpose(lAbe,((2,4),(1,3)))*ac
        end

        or(e,rightind::Int) = rightind == 0 ? 0 : or(e,re[rightind])
        function or(e,r::AbstractTensorMap)
            ar = ac*transpose(r,((1,),(3,2)))
            ear = e*transpose(ar,((2,4),(1,3)))
            return transpose(ear,((3,1),(4,2)))*ac_flipped
        end

        lr(l,r) = (l === 0 || r === 0) ? zero(Elt) : lr(l isa Int ? le[l] : l, r isa Int ? re[r] : r)
        lr(l::AbstractTensorMap,r::AbstractTensorMap) =
            tr(transpose(l,((),(3,2,1)))*transpose(r,((1,2,3),())))

        fast_expval(leftind,rightind,opp) = lr(lo(leftind,opp),rightind)

        

        # onsite
        let
            expv = fast_expval(1,r_end,add_util_leg(h_pm));
            @lock acc_lock dK[loc,loc] += expv
            for i in 1:half_basis_size-1, j in i+1:half_basis_size
                if i == loc
                    @lock acc_lock dV[j,i,j,i] -= expv
                    @lock acc_lock dV[i,j,i,j] -= expv
                end
            end
            for i in half_basis_size:basis_size, j in i+1:basis_size
                if j == loc
                    @lock acc_lock dV[i,j,i,j] -= expv
                    @lock acc_lock dV[j,i,j,i] -= expv
                end
            end
            for i in 1:half_basis_size-1,j in half_basis_size+1:basis_size
                if j == loc
                    @lock acc_lock dV[i,j,i,j] -= expv
                    @lock acc_lock dV[j,i,j,i] -= expv
                end
            end

            expv = fast_expval(1,r_end,add_util_leg(h_ppmm));
            @lock acc_lock dV[loc,loc,loc,loc] += expv
        end    
   

        # ---
        @sync begin

            let
                r_1 = or(bp_ut,r_end);
                r_2 = or(bm_ut,r_end);
                r_3 = or(ppLm,r_end);
                r_4 = or(Lpmm,r_end);
                for i in 1:loc-1
                    @Threads.spawn begin
                        @lock acc_lock dK[loc,i] += lr(indmap_1L[2,i],r_1)
                        @lock acc_lock dK[i,loc] += lr(indmap_1L[1,i],r_2)

                        expv = lr(indmap_1L[2,i],r_3);
                        @lock acc_lock dV[loc,loc,i,loc] += expv
                        @lock acc_lock dV[loc,loc,loc,i] += expv

                        expv = lr(indmap_1L[1,i],r_4);
                        @lock acc_lock dV[loc,i,loc,loc] += expv
                        @lock acc_lock dV[i,loc,loc,loc] += expv
                    end
                end
            end

            # ---

            let
                l_1 = lo(1,ppRm);
                l_2 = lo(1,Rpmm);
                for j in loc+1:basis_size      
                    expv = lr(l_1,indmap_1R[2,j]);
                    @lock acc_lock dV[loc,loc,j,loc] += expv
                    @lock acc_lock dV[loc,loc,loc,j] += expv

                    expv = lr(l_2,indmap_1R[1,j]);
                    @lock acc_lock dV[j,loc,loc,loc] += expv
                    @lock acc_lock dV[loc,j,loc,loc] += expv
                end
            end    
            # ---

            # 1 2 1
            for i in 1:half_basis_size,j in i+1:half_basis_size-1
                j == loc || continue;

                @Threads.spawn begin
                
                    l_1 = lo(indmap_1L[1,i],LRmm);
                    l_2 = lo(indmap_1L[2,i],ppLR);
                    l_3 = lo(indmap_1L[1,i],LpRm);
                    l_4 = lo(indmap_1L[1,i],p_pm_p)
                    l_5 = lo(indmap_1L[2,i],RpLm)
                    l_6 = lo(indmap_1L[2,i],m_pm_m)

                    for k in j+1:basis_size
                        expv = lr(l_1,indmap_1R[1,k])
                        @lock acc_lock dV[k,i,j,j] += expv
                        @lock acc_lock dV[i,k,j,j] += expv

                        expv = lr(l_2,indmap_1R[2,k]);
                        @lock acc_lock dV[j,j,k,i] += expv
                        @lock acc_lock dV[j,j,i,k] += expv

                        expv = lr(l_3,indmap_1R[2,k]);
                        @lock acc_lock dV[j,i,j,k] += expv
                        @lock acc_lock dV[i,j,k,j] += expv

                        expv = lr(l_4,indmap_1R[2,k]);
                        @lock acc_lock dV[j,i,k,j] += expv
                        @lock acc_lock dV[i,j,j,k] += expv

                        expv = lr(l_5,indmap_1R[1,k]);
                        @lock acc_lock dV[j,k,j,i] += expv
                        @lock acc_lock dV[k,j,i,j] += expv

                        expv = lr(l_6,indmap_1R[1,k]);
                        @lock acc_lock dV[j,k,i,j] += expv
                        @lock acc_lock dV[k,j,j,i] += expv
                    end
                end

            end

            # 2 1 1
            for i in 1:half_basis_size,j in i+1:half_basis_size
                j == loc || continue;

                @Threads.spawn begin
                    l_1 = lo(indmap_2L[1,i,1,i],bm_m);
                    l_2 = lo(indmap_2L[2,i,1,i],LpLR_1);
                    l_3 = lo(indmap_2L[2,i,1,i],bp_m);
                    l_4 = lo(indmap_2L[2,i,1,i],LRLm_1);
                    l_5 = lo(indmap_2L[2,i,1,i],bm_p);
                    l_6 = lo(indmap_2L[2,i,2,i],RpLL);

                    for k in j+1:basis_size

                        expv = lr(l_1,indmap_1R[2,k]);
                        @lock acc_lock dV[i,i,j,k] += expv
                        @lock acc_lock dV[i,i,k,j] += expv
                        
                        expv = lr(l_2,indmap_1R[2,k]);
                        @lock acc_lock dV[j,i,i,k] -= 2*expv
                        @lock acc_lock dV[i,j,k,i] -= 2*expv

                        expv = lr(l_3,indmap_1R[2,k]);
                        @lock acc_lock dV[i,j,i,k] += expv
                        @lock acc_lock dV[j,i,k,i] += expv

                        expv = lr(l_4,indmap_1R[1,k]);
                        @lock acc_lock dV[i,k,i,j] += 2*expv
                        @lock acc_lock dV[k,i,j,i] += 2*expv
                        @lock acc_lock dV[i,k,j,i] -= 2*expv
                        @lock acc_lock dV[k,i,i,j] -= 2*expv
                        
                        expv = lr(l_5,indmap_1R[1,k]);
                        @lock acc_lock dV[i,k,i,j] -= expv
                        @lock acc_lock dV[k,i,j,i] -= expv

                        expv = lr(l_6,indmap_1R[1,k]);
                        @lock acc_lock dV[j,k,i,i] += expv
                        @lock acc_lock dV[k,j,i,i] += expv
                    end
                end
            end


            # 1 2 1
            for j in half_basis_size:basis_size, k in j+1:basis_size
                j == loc || continue;

                @Threads.spawn begin
                    r_1 = or(LRmm,indmap_1R[1,k]);
                    r_2 = or(ppLR,indmap_1R[2,k]);
                    r_3 = or(LpRm,indmap_1R[2,k]);
                    r_4 = or(p_pm_p,indmap_1R[2,k]);
                    r_5 = or(RpLm,indmap_1R[1,k]);
                    r_6 = or(m_pm_m,indmap_1R[1,k]);

                    for i in 1:j-1
                        expv = lr(indmap_1L[1,i],r_1);
                        @lock acc_lock dV[k,i,j,j]+=expv
                        @lock acc_lock dV[i,k,j,j]+=expv

                        expv = lr(indmap_1L[2,i],r_2);
                        @lock acc_lock dV[j,j,k,i]+=expv
                        @lock acc_lock dV[j,j,i,k]+=expv

                        expv = lr(indmap_1L[1,i],r_3);
                        @lock acc_lock dV[j,i,j,k]+=expv
                        @lock acc_lock dV[i,j,k,j]+=expv

                        expv = lr(indmap_1L[1,i],r_4);
                        @lock acc_lock dV[j,i,k,j]+=expv
                        @lock acc_lock dV[i,j,j,k]+=expv

                        expv = lr(indmap_1L[2,i],r_5);
                        @lock acc_lock dV[j,k,j,i]+=expv
                        @lock acc_lock dV[k,j,i,j]+=expv

                        expv = lr(indmap_1L[2,i],r_6);
                        @lock acc_lock dV[j,k,i,j]+=expv
                        @lock acc_lock dV[k,j,j,i]+=expv

                    end
                end
            end 

            # 1 1 2
            for j in half_basis_size:basis_size, k in j+1:basis_size
                j == loc || continue;
                @Threads.spawn begin
                    r_1 = or(jimm,indmap_2R[2,k,2,k]);
                    r_2 = or(ppji,indmap_2R[1,k,1,k]);
                    r_3 = or(jpim_1,indmap_2R[2,k,1,k]);
                    r_4 = or(jpim_1-m_ap,indmap_2R[2,k,1,k]);
                    r_5 = or(ipjm_1,indmap_2R[2,k,1,k]);
                    r_6 = or(p_am - ipjm_1,indmap_2R[2,k,1,k]);

                    for i in 1:j-1
                        expv = lr(indmap_1L[1,i],r_1)
                        @lock acc_lock dV[j,i,k,k] +=expv
                        @lock acc_lock dV[i,j,k,k] +=expv

                        expv = lr(indmap_1L[2,i],r_2);
                        @lock acc_lock dV[k,k,j,i]+=expv
                        @lock acc_lock dV[k,k,i,j]+=expv

                        expv = lr(indmap_1L[2,i],r_3);
                        @lock acc_lock dV[j,k,i,k]+=expv
                        @lock acc_lock dV[k,j,k,i]+=expv
                        @lock acc_lock dV[k,j,i,k]-=2*expv
                        @lock acc_lock dV[j,k,k,i]-=2*expv

                        expv = lr(indmap_1L[2,i],r_4);
                        @lock acc_lock dV[j,k,i,k]+=expv
                        @lock acc_lock dV[k,j,k,i]+=expv

                        expv = lr(indmap_1L[1,i],r_5);
                        @lock acc_lock dV[i,k,j,k]+=expv
                        @lock acc_lock dV[k,i,k,j]+=expv
                        @lock acc_lock dV[i,k,k,j]-=2*expv
                        @lock acc_lock dV[k,i,j,k]-=2*expv
                        
                        expv = lr(indmap_1L[1,i],r_6);
                        @lock acc_lock dV[i,k,j,k]+=expv
                        @lock acc_lock dV[k,i,k,j]+=expv
                    
                    end
                end

            end
            
            

            # 1 1 2 + 2 2
            for k in 2:half_basis_size
                k == loc || continue;

                @Threads.spawn begin
                    r_1 = or(bmbm_ut,r_end);
                    r_2 = or(bpbp_ut,r_end);
                    r_3 = or(bpbm_ut,r_end);
                    r_4 = or(_pm_left,r_end);
                    r_5 = or(bmbp_ut,r_end);
                    for i in 1:min(half_basis_size-1,k-1)
                        @Threads.spawn begin
                            @lock acc_lock dV[i,i,k,k] += lr(indmap_2L[1,i,1,i],r_1)
                            @lock acc_lock dV[k,k,i,i] += lr(indmap_2L[2,i,2,i],r_2)
                            
                            expv = lr(indmap_2L[2,i,1,i],r_3);
                            @lock acc_lock dV[k,i,k,i]+=expv
                            @lock acc_lock dV[i,k,i,k]+=expv

                            expv = lr(indmap_2L[2,i,1,i],r_4);
                            @lock acc_lock dV[k,i,i,k]+=expv
                            @lock acc_lock dV[i,k,k,i]+=expv
                        end     
                    end

                    for i in 1:k-1,j in i+1:k-1
                        @Threads.spawn begin
                            expv = lr(indmap_2L[1,i,1,j],r_1);
                            @lock acc_lock dV[i,j,k,k]+=expv
                            @lock acc_lock dV[j,i,k,k]+=expv

                            expv = lr(indmap_2L[2,i,2,j],r_2);
                            @lock acc_lock dV[k,k,j,i]+=expv
                            @lock acc_lock dV[k,k,i,j]+=expv

                            expv = lr(indmap_2L[2,i,1,j],r_4);
                            @lock acc_lock dV[k,j,k,i]-=expv
                            @lock acc_lock dV[j,k,i,k]-=expv
                            @lock acc_lock dV[j,k,k,i]+=expv
                            @lock acc_lock dV[k,j,i,k]+=expv

                            expv = lr(indmap_2L[1,i,2,j],r_4);
                            @lock acc_lock dV[i,k,k,j]+=expv
                            @lock acc_lock dV[k,i,j,k]+=expv

                            expv = lr(indmap_2L[1,i,2,j],r_5);
                            @lock acc_lock dV[k,i,k,j]+=expv
                            @lock acc_lock dV[i,k,j,k]+=expv

                            expv = lr(indmap_2L[2,i,1,j],r_5);
                            @lock acc_lock dV[k,j,k,i]-=expv
                            @lock acc_lock dV[j,k,i,k]-=expv
                        end
                    end
                end
            end
            
            
            # 2 1 1 + 2 2
            for i in half_basis_size:basis_size
                i == loc || continue;
                @Threads.spawn begin
                    l_1  = lo(1,ut_amam);
                    l_2 = lo(1,ut_apap)
                    l_3 = lo(1,ut_apam)
                    l_4 = lo(1,_pm_right)
                    l_5 = lo(1,ut_amap);
                    for j in i+1:basis_size
                        @Threads.spawn begin
                            @lock acc_lock dV[j,j,i,i] += lr(l_1,indmap_2R[1,j,1,j])
                            @lock acc_lock dV[i,i,j,j] += lr(l_2,indmap_2R[2,j,2,j])

                            expv = lr(l_3,indmap_2R[2,j,1,j]);
                            @lock acc_lock dV[i,j,i,j]+=expv
                            @lock acc_lock dV[j,i,j,i]+=expv

                            expv = lr(l_4,indmap_2R[2,j,1,j]);
                            @lock acc_lock dV[j,i,i,j] += expv
                            @lock acc_lock dV[i,j,j,i]+=expv
                        end
                    end
                    for j in i+1:basis_size,k in j+1:basis_size
                        @Threads.spawn begin
                            expv = lr(l_1,indmap_2R[1,j,1,k])
                            @lock acc_lock dV[j,k,i,i] += expv
                            @lock acc_lock dV[k,j,i,i] += expv

                            expv = lr(l_2,indmap_2R[2,j,2,k]);
                            @lock acc_lock dV[i,i,j,k] += expv
                            @lock acc_lock dV[i,i,k,j] += expv

                            expv = lr(l_4,indmap_2R[2,j,1,k]);
                            @lock acc_lock dV[i,k,i,j] -= expv
                            @lock acc_lock dV[k,i,j,i] -= expv
                            @lock acc_lock dV[k,i,i,j] += expv
                            @lock acc_lock dV[i,k,j,i] += expv

                            expv = lr(l_4,indmap_2R[1,j,2,k]);
                            @lock acc_lock dV[i,j,k,i] += expv
                            @lock acc_lock dV[j,i,i,k] += expv

                            expv = lr(l_5,indmap_2R[1,j,2,k]);
                            @lock acc_lock dV[j,i,k,i] += expv
                            @lock acc_lock dV[i,j,i,k] += expv

                            expv = lr(l_5,indmap_2R[2,j,1,k]);
                            @lock acc_lock dV[i,k,i,j] -= expv
                            @lock acc_lock dV[k,i,j,i] -= expv
                        end
                    end
                end

            end


            # 3 left of half_basis_size
            for k in 3:half_basis_size,l in k+1:basis_size
                
                numblocks = 0
                for a in 1:k-1, b in a+1:k-1
                    numblocks += 1
                end
                numblocks > basis_size-k || continue
                numblocks == 0 && continue
                
                k == loc || continue;
                @Threads.spawn begin
                    r_1 = or(LpLR_1,indmap_1R[2,l]);
                    r_2 = or(bp_m,indmap_1R[2,l]);
                    r_3 = or(LRLm_1,indmap_1R[1,l])
                    r_4 = or(bm_p,indmap_1R[1,l])
                    r_5 = or(jimR_1,indmap_1R[2,l])
                    r_6 = or(bm_m,indmap_1R[2,l])
                    r_7 = or(RpLL,indmap_1R[1,l])
                    r_8 = or(bp_p,indmap_1R[1,l]);

                    for i in 1:k-1, j in i+1:k-1
                        @Threads.spawn begin

                            expv = lr(indmap_2L[2,i,1,j],r_1);

                            @lock acc_lock dV[j,k,l,i] -= 2*expv
                            @lock acc_lock dV[k,j,i,l] -= 2*expv

                            expv = lr(indmap_2L[1,i,2,j],r_1);

                            @lock acc_lock dV[i,k,l,j] -= 2*expv
                            @lock acc_lock dV[k,i,j,l] -= 2*expv
                            @lock acc_lock dV[k,i,l,j] += 2*expv
                            @lock acc_lock dV[i,k,j,l] += 2*expv

                            
                            expv = lr(indmap_2L[2,i,1,j],r_2);
                            @lock acc_lock dV[k,j,l,i] += expv
                            @lock acc_lock dV[j,k,i,l] += expv


                            expv = lr(indmap_2L[1,i,2,j],r_2);
                            @lock acc_lock dV[k,i,l,j] -= expv
                            @lock acc_lock dV[i,k,j,l] -= expv

                            expv = lr(indmap_2L[2,i,1,j],r_3)
                            @lock acc_lock dV[j,l,k,i]-=2*expv
                            @lock acc_lock dV[l,j,i,k]-=2*expv
                            @lock acc_lock dV[l,j,k,i]+=2*expv
                            @lock acc_lock dV[j,l,i,k]+=2*expv
                            
                            expv = lr(indmap_2L[1,i,2,j],r_3)
                            @lock acc_lock dV[i,l,k,j]-=2*expv
                            @lock acc_lock dV[l,i,j,k]-=2*expv


                            expv = lr(indmap_2L[2,i,1,j],r_4)
                            @lock acc_lock dV[l,j,k,i] -= expv
                            @lock acc_lock dV[j,l,i,k] -= expv
                        
                            expv = lr(indmap_2L[1,i,2,j],r_4)
                            @lock acc_lock dV[l,i,k,j] += expv
                            @lock acc_lock dV[i,l,j,k] += expv
                            
                            expv = lr(indmap_2L[1,i,1,j],r_5)
                            @lock acc_lock dV[j,i,l,k] += 2*expv
                            @lock acc_lock dV[i,j,k,l] += 2*expv


                            expv = lr(indmap_2L[1,i,1,j],r_6)
                            @lock acc_lock dV[i,j,l,k] += expv
                            @lock acc_lock dV[j,i,k,l] += expv
                            @lock acc_lock dV[j,i,l,k] -= expv
                            @lock acc_lock dV[i,j,k,l] -= expv

                            expv = lr(indmap_2L[2,i,2,j],r_7)
                            @lock acc_lock dV[l,k,j,i] += expv
                            @lock acc_lock dV[k,l,i,j] += expv

                            expv = lr(indmap_2L[2,i,2,j],r_8)
                            @lock acc_lock dV[l,k,i,j] += expv
                            @lock acc_lock dV[k,l,j,i] += expv
                        end
                    end
                end

            end
            
            
            for k in 3:half_basis_size, i in 1:k-1, j in i+1:k-1
                numblocks = 0
                for a in 1:k-1, b in a+1:k-1
                    numblocks += 1
                end
                numblocks <= basis_size-k || continue
                numblocks == 0 && continue

                k == loc || continue;

                @Threads.spawn begin
                    l_1 = lo(indmap_2L[2,i,1,j],LpLR_1)
                    l_2 = lo(indmap_2L[1,i,2,j],LpLR_1)
                    
                    l_3 = lo(indmap_2L[2,i,1,j],bp_m)
                    l_4 = lo(indmap_2L[1,i,2,j],bp_m)

                    l_5 = lo(indmap_2L[2,i,1,j],LRLm_1)
                    l_6 = lo(indmap_2L[2,i,1,j],bm_p)
                    l_7 = lo(indmap_2L[1,i,2,j],LRLm_1)
                    l_8 = lo(indmap_2L[1,i,2,j],bm_p)

                    l_9 = lo(indmap_2L[1,i,1,j],jimR_1)
                    l_10 = lo(indmap_2L[1,i,1,j],bm_m)

                    l_11 = lo(indmap_2L[2,i,2,j],RpLL)
                    l_12 = lo(indmap_2L[2,i,2,j],bp_p)


                    for l in k+1:basis_size
                        @Threads.spawn begin

                            expv = lr(l_1,indmap_1R[2,l]);
                            @lock acc_lock dV[j,k,l,i]-=2*expv
                            @lock acc_lock dV[k,j,i,l]-=2*expv
                        
                            expv = lr(l_2,indmap_1R[2,l]);
                            @lock acc_lock dV[i,k,l,j]-=2*expv
                            @lock acc_lock dV[k,i,j,l]-=2*expv
                            @lock acc_lock dV[k,i,l,j]+=2*expv
                            @lock acc_lock dV[i,k,j,l]+=2*expv

                            expv = lr(l_3,indmap_1R[2,l]);
                            @lock acc_lock dV[k,j,l,i]+=expv
                            @lock acc_lock dV[j,k,i,l]+=expv

                            expv = lr(l_4,indmap_1R[2,l]);
                            @lock acc_lock dV[k,i,l,j]-=expv
                            @lock acc_lock dV[i,k,j,l]-=expv


                            expv = lr(l_5,indmap_1R[1,l]);
                            @lock acc_lock dV[j,l,k,i]-=2*expv
                            @lock acc_lock dV[l,j,i,k]-=2*expv
                            @lock acc_lock dV[l,j,k,i]+=2*expv
                            @lock acc_lock dV[j,l,i,k]+=2*expv

                            expv = lr(l_6,indmap_1R[1,l]);
                            @lock acc_lock dV[l,j,k,i]-=expv
                            @lock acc_lock dV[j,l,i,k]-=expv

                            expv = lr(l_7,indmap_1R[1,l]);
                            @lock acc_lock dV[i,l,k,j]-=2*expv
                            @lock acc_lock dV[l,i,j,k]-=2*expv


                            expv = lr(l_8,indmap_1R[1,l]);
                            @lock acc_lock dV[l,i,k,j]+=expv
                            @lock acc_lock dV[i,l,j,k]+=expv


                            expv = lr(l_9,indmap_1R[2,l]);
                            @lock acc_lock dV[j,i,l,k]+=2*expv
                            @lock acc_lock dV[i,j,k,l]+=2*expv

                            expv = lr(l_10,indmap_1R[2,l]);
                            @lock acc_lock dV[i,j,l,k]+=expv
                            @lock acc_lock dV[j,i,k,l]+=expv
                            @lock acc_lock dV[j,i,l,k]-=expv
                            @lock acc_lock dV[i,j,k,l]-=expv

                            expv = lr(l_11,indmap_1R[1,l]);
                            @lock acc_lock dV[l,k,j,i]+=expv
                            @lock acc_lock dV[k,l,i,j]+=expv


                            expv = lr(l_12,indmap_1R[1,l]);
                            @lock acc_lock dV[l,k,i,j]+=expv
                            @lock acc_lock dV[k,l,j,i]+=expv
                        end
                    end
                end
            end
            
            # 3 right of half_basis_size
            for i in 1:basis_size,j in i+1:basis_size
                j >= half_basis_size || continue
                
                #=
                numblocks = 0
                for a in max(j+1,half_basis_size+1):basis_size, b in a+1:basis_size
                    numblocks += 1
                end
                numblocks/8 > (j-1)/12 || continue
                numblocks == 0 && continue
                =#
                j == loc || continue;
                @Threads.spawn begin
                    l_1 = lo(indmap_1L[2,i],-2*jpim_1);
                    l_2 = lo(indmap_1L[2,i],m_ap-jpim_1);
                    l_3 = lo(indmap_1L[1,i],-2*ipjm_1);
                    l_4 = lo(indmap_1L[1,i],-(p_am-ipjm_1));
                    l_5 = lo(indmap_1L[2,i],(m_am + ppji)/2)
                    l_6 = lo(indmap_1L[2,i],(m_am - ppji)/2)
                    l_7 = lo(indmap_1L[1,i],(p_ap+jimm)/2)
                    l_8 = lo(indmap_1L[1,i],(p_ap-jimm)/2)

                    for k in max(j+1,half_basis_size+1):basis_size,l in k+1:basis_size
                        @Threads.spawn begin

                            expv = lr(l_1,indmap_2R[1,k,2,l]);
                            @lock acc_lock dV[j,k,i,l]-=expv/2
                            @lock acc_lock dV[k,j,l,i]-=expv/2
                            @lock acc_lock dV[k,j,i,l]+=expv
                            @lock acc_lock dV[j,k,l,i]+=expv

                            expv = lr(l_1,indmap_2R[2,k,1,l]);
                            @lock acc_lock dV[l,j,k,i]-=expv/2
                            @lock acc_lock dV[j,l,i,k]-=expv/2
                            @lock acc_lock dV[l,j,i,k]+=expv
                            @lock acc_lock dV[j,l,k,i]+=expv

                            expv = lr(l_2,indmap_2R[1,k,2,l]);
                            @lock acc_lock dV[j,k,i,l]+=expv
                            @lock acc_lock dV[k,j,l,i]+=expv
                            
                            expv = lr(l_2,indmap_2R[2,k,1,l]);
                            @lock acc_lock dV[l,j,k,i]-=expv
                            @lock acc_lock dV[j,l,i,k]-=expv

                            expv = lr(l_3,indmap_2R[1,k,2,l],);
                            @lock acc_lock dV[i,k,j,l]-=expv/2
                            @lock acc_lock dV[k,i,l,j]-=expv/2
                            @lock acc_lock dV[k,i,j,l]+=expv
                            @lock acc_lock dV[i,k,l,j]+=expv

                            expv = lr(l_3,indmap_2R[2,k,1,l]);
                            @lock acc_lock dV[l,i,k,j]-=expv/2
                            @lock acc_lock dV[i,l,j,k]-=expv/2
                            @lock acc_lock dV[l,i,j,k]+=expv
                            @lock acc_lock dV[i,l,k,j]+=expv

                            expv = lr(l_4,indmap_2R[1,k,2,l]);
                            @lock acc_lock dV[i,k,j,l]+=expv
                            @lock acc_lock dV[k,i,l,j]+=expv

                            expv = lr(l_4,indmap_2R[2,k,1,l]);
                            @lock acc_lock dV[l,i,k,j]-=expv
                            @lock acc_lock dV[i,l,j,k]-=expv

                            expv = lr(l_5,indmap_2R[1,k,1,l]);
                            @lock acc_lock dV[l,k,j,i]+=expv
                            @lock acc_lock dV[k,l,i,j]+=expv
                            @lock acc_lock dV[l,k,i,j]+=expv
                            @lock acc_lock dV[k,l,j,i]+=expv

                            expv = lr(l_6,indmap_2R[1,k,1,l]);
                            @lock acc_lock dV[l,k,j,i]-=expv
                            @lock acc_lock dV[k,l,i,j]-=expv
                            @lock acc_lock dV[l,k,i,j]+=expv
                            @lock acc_lock dV[k,l,j,i]+=expv

                            expv = lr(l_7,indmap_2R[2,k,2,l]);
                            @lock acc_lock dV[j,i,k,l]+=expv
                            @lock acc_lock dV[i,j,l,k]+=expv
                            @lock acc_lock dV[i,j,k,l]+=expv
                            @lock acc_lock dV[j,i,l,k]+=expv

                            expv = lr(l_8,indmap_2R[2,k,2,l]);
                            @lock acc_lock dV[j,i,k,l]+=expv
                            @lock acc_lock dV[i,j,l,k]+=expv
                            @lock acc_lock dV[i,j,k,l]-=expv
                            @lock acc_lock dV[j,i,l,k]-=expv
                        end

                    end
                end

            end
                        

             # loc == half_basis_size: 
            for k in half_basis_size+1:basis_size, l in k+1:basis_size
                loc == half_basis_size || continue;

                @Threads.spawn begin
                    r_1 = or(pm_ai_pm,indmap_2R[1,k,2,l]);
                    r_2 = or(pm_ai_pm,indmap_2R[2,k,1,l]);
                    r_3 = or(pp_ai_pp,indmap_2R[2,k,2,l]);
                    r_4 = or(mm_ai_mm,indmap_2R[1,k,1,l]);
                    r_5 = or(jkil_2,indmap_2R[1,k,2,l])
                    r_6 = or(jkil_2,indmap_2R[2,k,1,l])
                    r_7 = or(jikl_1,indmap_2R[2,k,2,l])
                    r_8 = or(lkij_1,indmap_2R[1,k,1,l])
                    
                    for i in 1:half_basis_size-1,j in i+1:half_basis_size-1
                        @Threads.spawn begin
                            expv = lr(indmap_2L[2,i,1,j],r_1);
                            @lock acc_lock dV[j,k,i,l]+=expv
                            @lock acc_lock dV[k,j,l,i]+=expv
                            @lock acc_lock dV[k,j,i,l]-=2*expv
                            @lock acc_lock dV[j,k,l,i]-=2*expv

                            expv = lr(indmap_2L[1,i,2,j],r_1);
                            @lock acc_lock dV[k,i,j,l]-=2*expv
                            @lock acc_lock dV[i,k,l,j]-=2*expv
                            @lock acc_lock dV[i,k,j,l]+=expv
                            @lock acc_lock dV[k,i,l,j]+=expv

                            expv = lr(indmap_2L[1,i,2,j],r_2);
                            @lock acc_lock dV[l,i,k,j]+=expv
                            @lock acc_lock dV[i,l,j,k]+=expv
                            @lock acc_lock dV[l,i,j,k]-=2*expv
                            @lock acc_lock dV[i,l,k,j]-=2*expv

                            expv = lr(indmap_2L[2,i,1,j],r_2);
                            @lock acc_lock dV[l,j,k,i]+=expv
                            @lock acc_lock dV[j,l,i,k]+=expv
                            @lock acc_lock dV[l,j,i,k]-=2*expv
                            @lock acc_lock dV[j,l,k,i]-=2*expv


                            expv = lr(indmap_2L[1,i,1,j],r_3);
                            @lock acc_lock dV[j,i,k,l]+=expv
                            @lock acc_lock dV[i,j,l,k]+=expv
                            @lock acc_lock dV[i,j,k,l]-=expv
                            @lock acc_lock dV[j,i,l,k]-=expv

                            expv = lr(indmap_2L[2,i,2,j],r_4);
                            @lock acc_lock dV[l,k,i,j]+=expv
                            @lock acc_lock dV[k,l,j,i]+=expv
                            @lock acc_lock dV[l,k,j,i]-=expv
                            @lock acc_lock dV[k,l,i,j]-=expv


                            expv = lr(indmap_2L[2,i,1,j],r_5);
                            @lock acc_lock dV[k,j,i,l]+=2*expv
                            @lock acc_lock dV[j,k,l,i]+=2*expv

                            expv = lr(indmap_2L[1,i,2,j],r_5);
                            @lock acc_lock dV[k,i,j,l]+=2*expv
                            @lock acc_lock dV[i,k,l,j]+=2*expv
                            @lock acc_lock dV[i,k,j,l]-=2*expv
                            @lock acc_lock dV[k,i,l,j]-=2*expv

                            expv = lr(indmap_2L[1,i,2,j],r_6);
                            @lock acc_lock dV[l,i,j,k]+=2*expv
                            @lock acc_lock dV[i,l,k,j]+=2*expv

                            expv = lr(indmap_2L[2,i,1,j],r_6);
                            @lock acc_lock dV[l,j,k,i]-=2*expv
                            @lock acc_lock dV[j,l,i,k]-=2*expv
                            @lock acc_lock dV[l,j,i,k]+=2*expv
                            @lock acc_lock dV[j,l,k,i]+=2*expv

                            expv = lr(indmap_2L[1,i,1,j],r_7);
                            @lock acc_lock dV[i,j,k,l]+=2*expv
                            @lock acc_lock dV[j,i,l,k]+=2*expv

                            expv = lr(indmap_2L[2,i,2,j],r_8);
                            @lock acc_lock dV[l,k,j,i]+=2*expv
                            @lock acc_lock dV[k,l,i,j]+=2*expv
                        end

                    end
                end
            end
            
            for k in half_basis_size+1:basis_size
                loc == half_basis_size || continue;

                @Threads.spawn begin
                    r_1 = or(pm_ai_pm,indmap_2R[2,k,1,k])
                    r_2 = or(jkil_2,indmap_2R[2,k,1,k])
                    r_3 = or(mm_ai_mm,indmap_2R[1,k,1,k])
                    r_4 = or(pp_ai_pp,indmap_2R[2,k,2,k])
                    for i in 1:half_basis_size-1
                        @Threads.spawn begin
                            expv = lr(indmap_2L[1,i,2,i],r_1);
                            @lock acc_lock dV[i,k,i,k]+=expv
                            @lock acc_lock dV[k,i,k,i]+=expv

                            expv = lr(indmap_2L[2,i,1,i],r_1);
                            @lock acc_lock dV[i,k,k,i]-=2*expv
                            @lock acc_lock dV[k,i,i,k]-=2*expv

                            expv = lr(indmap_2L[2,i,1,i],r_2);
                            @lock acc_lock dV[i,k,k,i]+=2*expv
                            @lock acc_lock dV[k,i,i,k]+=2*expv

                            expv = lr(indmap_2L[2,i,2,i],r_3);
                            @lock acc_lock dV[k,k,i,i]+=expv

                            expv = lr(indmap_2L[1,i,1,i],r_4);
                            @lock acc_lock dV[i,i,k,k]+=expv
                        end
                    end

                    for i in 1:half_basis_size-1, j in i+1:half_basis_size-1
                        @Threads.spawn begin

                            expv = lr(indmap_2L[2,i,1,j],r_1);
                            @lock acc_lock dV[j,k,i,k]+=expv
                            @lock acc_lock dV[k,j,k,i]+=expv
                            @lock acc_lock dV[k,j,i,k]-=2*expv
                            @lock acc_lock dV[j,k,k,i]-=2*expv

                            expv = lr(indmap_2L[1,i,2,j],r_1);
                            @lock acc_lock dV[i,k,j,k]+=expv
                            @lock acc_lock dV[k,i,k,j]+=expv
                            @lock acc_lock dV[i,k,k,j]-=2*expv
                            @lock acc_lock dV[k,i,j,k]-=2*expv

                            expv = lr(indmap_2L[2,i,1,j],r_2);
                            @lock acc_lock dV[j,k,i,k]-=2*expv
                            @lock acc_lock dV[k,j,k,i]-=2*expv
                            @lock acc_lock dV[k,j,i,k]+=2*expv
                            @lock acc_lock dV[j,k,k,i]+=2*expv

                            expv = lr(indmap_2L[1,i,2,j],r_2);
                            @lock acc_lock dV[i,k,k,j]+=2*expv
                            @lock acc_lock dV[k,i,j,k]+=2*expv

                            expv = lr(indmap_2L[2,i,2,j],r_3);
                            @lock acc_lock dV[k,k,j,i]+=expv
                            @lock acc_lock dV[k,k,i,j]+=expv

                            expv = lr(indmap_2L[1,i,1,j],r_4);
                            @lock acc_lock dV[j,i,k,k]+=expv
                            @lock acc_lock dV[i,j,k,k]+=expv
                        end
                    end
                end    
            end
            
            for i in 1:half_basis_size-1
                loc == half_basis_size || continue;

                @Threads.spawn begin

                    l_1 = lo(indmap_2L[2,i,2,i],mm_ai_mm);
                    l_2 = lo(indmap_2L[1,i,1,i],pp_ai_pp);
                    l_3 = lo(indmap_2L[2,i,1,i],pm_ai_pm);
                    l_4 = lo(indmap_2L[2,i,1,i],jkil_2);
                    for j in half_basis_size+1:basis_size,k in j+1:basis_size
                        @Threads.spawn begin

                            expv = lr(l_1,indmap_2R[1,j,1,k]);
                            @lock acc_lock dV[j,k,i,i]+=expv
                            @lock acc_lock dV[k,j,i,i]+=expv

                            expv = lr(l_2,indmap_2R[2,j,2,k]);
                            @lock acc_lock dV[i,i,j,k]+=expv
                            @lock acc_lock dV[i,i,k,j]+=expv

                            expv = lr(l_3,indmap_2R[1,j,2,k]);
                            @lock acc_lock dV[j,i,i,k]-=2*expv
                            @lock acc_lock dV[i,j,k,i]-=2*expv
                            @lock acc_lock dV[i,j,i,k]+=expv
                            @lock acc_lock dV[j,i,k,i]+=expv

                            expv = lr(l_3,indmap_2R[2,j,1,k]);
                            @lock acc_lock dV[k,i,i,j]-=2*expv
                            @lock acc_lock dV[i,k,j,i]-=2*expv
                            @lock acc_lock dV[k,i,j,i]+=expv
                            @lock acc_lock dV[i,k,i,j]+=expv

                            expv = lr(l_4,indmap_2R[1,j,2,k]);
                            @lock acc_lock dV[j,i,i,k]+=2*expv
                            @lock acc_lock dV[i,j,k,i]+=2*expv

                            expv = lr(l_4,indmap_2R[2,j,1,k]);
                            @lock acc_lock dV[k,i,i,j]+=2*expv
                            @lock acc_lock dV[i,k,j,i]+=2*expv
                            @lock acc_lock dV[k,i,j,i]-=2*expv
                            @lock acc_lock dV[i,k,i,j]-=2*expv

                        end
                    end
                end

            end
        
        end
        
        
       
    end
    #@show energy
    (dV,dK)
end
# the rdms only use the pure operator channels (a_i, a_i a_j, ...) of the mpo, whose environments do not depend
# on the integrals. compress drops channels that only couple to zero integrals, so we build the mpo with dense
# integrals to make sure no rdm element goes missing.
const _rdm_hamiltonians = Dict{Int,Any}()
const _rdm_hamiltonians_lock = ReentrantLock()
function rdm_hamiltonian(N::Int)
    @lock _rdm_hamiltonians_lock get!(_rdm_hamiltonians,N) do
        (ham,indmaps...) = fused_quantum_chemistry_hamiltonian(0.0,ones(N,N),ones(N,N,N,N),Float64)
        (ham,indmaps)
    end
end

"""
    qchem_rdms(state) -> (dV,dK)

one and two body reduced density matrices of `state`, such that the energy of the hamiltonian
`fused_quantum_chemistry_hamiltonian(E0,K,V)` is `E0 + sum(K.*dK) + sum(V.*dV)`
"""
function qchem_rdms(state)
    (ham,indmaps) = rdm_hamiltonian(length(state))
    (dV,dK) = quantum_chemistry_dV_dK(state,ham,indmaps)
    (real(dV),real(dK))
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

qchem_mpo(ints::QChemIntegrals) = first(fused_quantum_chemistry_hamiltonian(ints.E0,ints.K,ints.V,Float64))

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
