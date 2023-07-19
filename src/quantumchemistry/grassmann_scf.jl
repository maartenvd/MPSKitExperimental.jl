#=
The chemistry hamiltonian is often treated by freezing a set of orbitals to be frozen-filled, another set to be frozen-empty, and the rest using an MPS
This structure represents this hamiltonian (the full ERI,K,Vnuc, along with the set of orbitals that are actively treated)
=#

struct GrassmannSCF <: MPSKit.Algorithm
    maxiter::Int
    tol::Float64
end

GrassmannSCF(;tol = 1e-12,maxiter=100) = GrassmannSCF(maxiter,tol);



struct CASSCF_Ham{A,B,C,D,E}
    ERI::A
    K::B
    Vnuc::C
    U::D
    active::E
end

function MPSKit.find_groundstate(state,ham::CASSCF_Ham,alg::GrassmannSCF)


    (out,fx,_,_,normgradhistory) = optimize(cfun,manifoldpoint(state,ham), ConjugateGradient(gradtol=alg.tol, maxiter=alg.maxiter, verbosity = 2);
                retract = retract, inner = inner, transport! = transport! ,
                scale! = scale! , add! = add! , isometrictransport = true,precondition = precondition);
    st = out[1]; ham = out[2];
    return (st,ham)
end

function quantum_chemistry_dV_dK(ham::CASSCF_Ham,state)
    (qchemham,indmap_1Ls, indmap_1Rs, indmap_2Ls, indmap_2Rs) = mpo_representation(ham)
    envs = disk_environments(state,qchemham)
    
    basis_size = length(state);

    half_basis_size = Int(ceil((basis_size+1)/2));
    
    Elt = eltype(state.AL[1])

    dK = fill(zero(Elt),basis_size,basis_size);
    dV = fill(zero(Elt),basis_size,basis_size,basis_size,basis_size);

    psp = Vect[(Irrep[U₁]⊠Irrep[SU₂] ⊠ FermionParity)]((0,0,0)=>1, (1,1//2,1)=>1, (2,0,0)=>1);

    ap = TensorMap(ones,Elt,psp*Vect[(Irrep[U₁]⊠Irrep[SU₂] ⊠ FermionParity)]((-1,1//2,1)=>1),psp);
    blocks(ap)[(U₁(0)⊠SU₂(0)⊠FermionParity(0))] .*= -sqrt(2);
    blocks(ap)[(U₁(1)⊠SU₂(1//2)⊠FermionParity(1))]  .*= 1;


    bm = TensorMap(ones,Elt,psp,Vect[(Irrep[U₁]⊠Irrep[SU₂]⊠FermionParity)]((-1,1//2,1)=>1)*psp);
    blocks(bm)[(U₁(0)⊠SU₂(0)⊠FermionParity(0))] .*= sqrt(2);
    blocks(bm)[(U₁(1)⊠SU₂(1//2)⊠FermionParity(1))] .*= -1;

    # this transposition is easier to reason about in a planar way
    am = transpose(ap',(2,1),(3,));
    bp = transpose(bm',(1,),(3,2));
    ap = transpose(ap,(3,1),(2,));
    bm = transpose(bm,(2,),(3,1));
    
    flipcor = isometry(flip(space(bm,1)),space(bm,1));
    bm = flipcor*bm;
    ap = ap*flipcor';

    @plansor b_derp[-1 -2;-3] := bp[1;2 -2]*τ[-3 -1;2 1]
    @plansor b_derp[-1 -2;-3] := bm[1;2 -2]*τ[-3 -1;2 1]

    h_pm = TensorMap(ones,Elt,psp,psp);
    blocks(h_pm)[(U₁(0)⊠SU₂(0)⊠ FermionParity(0))] .=0;
    blocks(h_pm)[(U₁(1)⊠SU₂(1//2)⊠ FermionParity(1))] .=1;
    blocks(h_pm)[(U₁(2)⊠SU₂(0)⊠ FermionParity(0))] .=2;

    @plansor o_derp[-1 -2;-3 -4] := am[-1 1;-3]*ap[1 -2;-4]
    h_pm_derp = transpose(h_pm,(2,1),());
    Lmap_apam_to_pm = find_right_map(o_derp,h_pm_derp)

    @plansor o_derp[-1 -2;-3 -4] := bm[-1;-3 1]*bp[-2;1 -4]
    h_pm_derp2 = transpose(h_pm,(),(2,1));
    Rmap_bpbm_to_pm = find_left_map(o_derp,h_pm_derp2)

    h_ppmm = h_pm*h_pm-h_pm;
    
    ai = isomorphism(storagetype(ap),psp,psp);

    ut = Tensor(ones,oneunit(psp));
    @plansor ut_ap[-1; -3 -4 -2] := ut[-1]*ap[-3 -2;-4];
    @plansor ut_am[-1; -3 -4 -2] := ut[-1]*am[-3 -2;-4];
    @plansor bp_ut[-1; -3 -4 -2] := bp[-1;-3 -2]*conj(ut[-4]);
    @plansor bm_ut[-1; -3 -4 -2] := bm[-1;-3 -2]*conj(ut[-4]);
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
    @plansor ut_apap[-1; -3 -4 -2] := ut[-1]*ap[-3 1;3]*ap[1 -2;4]*conj(pp_f[-4;3 4]);
    @plansor ut_amam[-1; -3 -4 -2] := ut[-1]*am[-3 1;3]*am[1 -2;4]*conj(mm_f[-4;3 4]);
    @plansor ut_amap[-1; -3 -4 -2] := ut[-1]*am[-3 1;3]*ap[1 -2;4]*conj(mp_f[-4;3 4]);
    @plansor ut_apam[-1; -3 -4 -2] := ut[-1]*ap[-3 1;3]*am[1 -2;4]*conj(pm_f[-4;3 4])
    @plansor bpbp_ut[-1; -3 -4 -2] := mm_f[-1;1 2]*bp[1;-3 3]*bp[2;3 -2]*conj(ut[-4]);
    @plansor bmbm_ut[-1; -3 -4 -2] := pp_f[-1;1 2]*bm[1;-3 3]*bm[2;3 -2]*conj(ut[-4]);
    @plansor bmbp_ut[-1; -3 -4 -2] := pm_f[-1;1 2]*bm[1;-3 3]*bp[2;3 -2]*conj(ut[-4]);
    @plansor bpbm_ut[-1; -3 -4 -2] := mp_f[-1;1 2]*bp[1;-3 3]*bm[2;3 -2]*conj(ut[-4]);
    iso_pp = isomorphism(_lastspace(ap)',_lastspace(ap)');
    iso_mm = isomorphism(_lastspace(am)',_lastspace(am)');
    @plansor p_ai_p[-1; -3 -4 -2] := iso_pp[-1;1]*τ[1 2;-3 -4]*ai[-2;2]
    @plansor m_ai_m[-1; -3 -4 -2] := iso_mm[-1;1]*τ[1 2;-3 -4]*ai[-2;2]
    @plansor p_pm_p[-1; -3 -4 -2] := iso_pp[-1;1]*τ[1 2;-3 -4]*h_pm[-2;2]
    @plansor m_pm_m[-1; -3 -4 -2] := iso_mm[-1;1]*τ[1 2;-3 -4]*h_pm[-2;2]
    iso_pppp = pp_f*pp_f';
    iso_pmpm = pm_f*pm_f';
    iso_mmmm = mm_f*mm_f';
    @plansor pp_ai_pp[-1; -3 -4 -2] := iso_pppp[-1;1]*τ[1 2;-3 -4]*ai[-2;2]
    @plansor pm_ai_pm[-1; -3 -4 -2] := iso_pmpm[-1;1]*τ[1 2;-3 -4]*ai[-2;2]
    @plansor mm_ai_mm[-1; -3 -4 -2] := iso_mmmm[-1;1]*τ[1 2;-3 -4]*ai[-2;2]
    @plansor p_ap[-1; -3 -4 -2] := iso_pp[-1;1]*τ[1 2;-3 3]*ap[2 -2;4]*conj(pp_f[-4;3 4]);
    @plansor m_ap[-1; -3 -4 -2] := iso_mm[-1;1]*τ[1 2;-3 3]*ap[2 -2;4]*conj(mp_f[-4;3 4]);
    @plansor p_am[-1; -3 -4 -2] := iso_pp[-1;1]*τ[1 2;-3 3]*am[2 -2;4]*conj(pm_f[-4;3 4]);
    @plansor m_am[-1; -3 -4 -2] := iso_mm[-1;1]*τ[1 2;-3 3]*am[2 -2;4]*conj(mm_f[-4;3 4]);
    @plansor bp_p[-1; -3 -4 -2] := bp[2;-3 3]*iso_mm[1;-4]*τ[4 -2;3 1]*mm_f[-1;2 4]
    @plansor bm_p[-1; -3 -4 -2] := bm[2;-3 3]*iso_mm[1;-4]*τ[4 -2;3 1]*pm_f[-1;2 4]
    @plansor bm_m[-1; -3 -4 -2] := bm[2;-3 3]*iso_pp[1;-4]*τ[4 -2;3 1]*pp_f[-1;2 4]
    @plansor bp_m[-1; -3 -4 -2] := bp[2;-3 3]*iso_pp[1;-4]*τ[4 -2;3 1]*mp_f[-1;2 4]
    @plansor ppLm[-1; -3 -4 -2] := bp[-1;1 -2]*h_pm[1;-3]*conj(ut[-4])
    @plansor Lpmm[-1; -3 -4 -2] := bm[-1;-3 1]*h_pm[-2;1]*conj(ut[-4])
    @plansor ppRm[-1; -3 -4 -2] := ut[-1]*ap[1 -2;-4]*h_pm[1;-3]
    @plansor Rpmm[-1; -3 -4 -2] := ut[-1]*h_pm[-2;1]*am[-3 1;-4]
    @plansor LRmm[-1; -3 -4 -2] := am[1 -2;-4]*bm[-1;-3 1]
    @plansor ppLR[-1; -3 -4 -2] := ap[1 -2;-4]*bp[-1;-3 1]
    @plansor LpRm[-1; -3 -4 -2] := ap[1 -2;-4]*bm[-1;-3 1]
    @plansor RpLm[-1; -3 -4 -2] := bp[-1;1 -2]*am[-3 1;-4]
    @plansor _pm_left[-1; -3 -4 -2] := (mp_f*Lmap_apam_to_pm)[-1]*h_pm[-2;-3]*conj(ut[-4])
    @plansor _pm_right[-1; -3 -4 -2] := ut[-1]*h_pm[-2;-3]*(transpose(Rmap_bpbm_to_pm*pm_f',(1,)))[-4]
    @plansor LRLm_1[-1; -3 -4 -2] := (mp_f_1)[-1;1 2]*bm[2;3 -2]*τ[1 3;-3 -4]
    @plansor LpLR_1[-1; -3 -4 -2] := (mp_f_1)[-1;1 2]*bp[1;-3 3]*τ[3 2;-4 -2]
    @plansor RpLL[-1; -3 -4 -2] := mm_f[-1;1 2]*bp[2;3 -2]*τ[1 3;-3 -4]
    @plansor jimm[-1; -3 -4 -2] := iso_pp[-1;1]*ap[-3 2;3]*τ[2 1;4 -2]*conj(pp_f[-4;3 4])
    @plansor ppji[-1; -3 -4 -2] := iso_mm[-1;1]*am[-3 2;3]*τ[2 1;4 -2]*conj(mm_f[-4;3 4])
    @plansor jpim_1[-1; -3 -4 -2] := iso_mm[-1;1]*ap[-3 2;3]*τ[2 1;4 -2]*conj((pm_f_1)[-4;3 4])
    @plansor ipjm_1[-1; -3 -4 -2] := iso_pp[-1;1]*τ[-3 1;2 3]*am[3 -2;4]*conj((pm_f_1)[-4;2 4])
    @plansor jimR_1[-1; -3 -4 -2] := pp_f_1[-1;1 2]*τ[3 2;-4 -2]*bm[1;-3 3]
    @plansor jkil_2[-1; -3 -4 -2] := mp_f_2[-1;1 2]*ai[3;-3]*τ[3 1;4 5]*τ[5 2;6 -2]*conj(mp_f_2[-4;4 6])
    @plansor jikl_1[-1; -3 -4 -2] := pp_f_1[-1;1 2]*ai[3;-3]*τ[3 1;4 5]*τ[5 2;6 -2]*conj(pp_f_1[-4;4 6])
    @plansor lkij_1[-1; -3 -4 -2] := mm_f_1[-1;1 2]*ai[3;-3]*τ[3 1;4 5]*τ[5 2;6 -2]*conj(mm_f_1[-4;4 6])

    dAC = similar.(state.AL);
    
    S = spacetype(eltype(state));
    storage = storagetype(eltype(state));

    mvaltype_2_3 = DelayedFactType(S,storage,2,3);
    mvaltype_3_1 = DelayedFactType(S,storage,3,1);
    tvaltype_2_1 = TransposeFactType(S,storage,2,1);
    tvaltype_3_2 = TransposeFactType(S,storage,3,2);
    tvaltype_2_2 = TransposeFactType(S,storage,2,2);

    
    mfactory_2_3 = Dict{Any,mvaltype_2_3}();
    mfactory_3_1 = Dict{Any,mvaltype_3_1}();
    tfactory_2_1 = Dict{Any,tvaltype_2_1}();
    tfactory_3_2 = Dict{Any,tvaltype_3_2}();
    tfactory_2_2 = Dict{Any,tvaltype_2_2}();

    le = leftenv(envs,1,state);
    re = rightenv(envs,1,state);
    ac = state.AC[1];

    function fast_expval(leftind,rightind,opp_transposed)
        if leftind == 0 || rightind == 0
            return zero(eltype(ac))
        end
        
        left = le[leftind];
        right = re[rightind];

        t_0_f = tfactory_2_1[(codomain(left)←domain(left),(3,1),(2,))];
        t_0 = t_0_f(left);

        t_1_f = mfactory_2_3[codomain(t_0)←domain(opp_transposed)]
        t_1 = t_1_f()
        mul!(t_1,t_0,opp_transposed);
        free!(t_0_f,t_0)

        t_2_f = tfactory_3_2[(codomain(t_1)←domain(t_1),(2,5,4),(1,3))];
        t_2 = t_2_f(t_1)
        free!(t_1_f,t_1);

        t_3_f = mfactory_3_1[codomain(t_2)←domain(ac)];
        t_3 = t_3_f();
        mul!(t_3,t_2,ac);
        free!(t_2_f,t_2);

        t_4_f = tfactory_2_2[(codomain(t_3)←domain(t_3),(1,2),(4,3))];
        t_4 = t_4_f(t_3);
        free!(t_3_f,t_3);

        t_5 = fast_similar(ac);
        mul!(t_5,t_4,right);
        free!(t_4_f,t_4);

        dot(ac,t_5)
    end

    for loc in 1:basis_size
        indmap_1L = indmap_1Ls[loc]
        indmap_2L = indmap_2Ls[loc]
        indmap_1R = indmap_1Rs[loc+1]
        indmap_2R = indmap_2Rs[loc+1]


        ac = state.AC[loc];
        le = leftenv(envs,loc,state);
        re = rightenv(envs,loc,state);
        l_end = length(le);
        r_end = length(re);

        (mfactory_2_3, mfactory_3_1, tfactory_2_1, tfactory_3_2, tfactory_2_2) = _make_AC_factories(qchemham[loc],ac);

        dAC[loc] = MPSKit.∂∂AC(loc, state, qchemham, envs)(ac);

        # onsite
        let
            expv = fast_expval(1,r_end,transpose(add_util_leg(h_pm),(1,),(3,4,2)));
            dK[loc,loc] += expv;
            for i in 1:half_basis_size-1, j in i+1:half_basis_size
                if i == loc
                    dV[j,i,j,i] -= expv;
                    dV[i,j,i,j] -= expv;
                end
            end
            for i in half_basis_size:basis_size, j in i+1:basis_size
                if j == loc
                    dV[i,j,i,j] -= expv;
                    dV[j,i,j,i] -= expv;
                end
            end
            for i in 1:half_basis_size-1,j in half_basis_size+1:basis_size
                if j == loc
                    dV[i,j,i,j] -= expv;
                    dV[j,i,j,i] -= expv;
                end
            end

            expv = fast_expval(1,r_end,transpose(add_util_leg(h_ppmm),(1,),(3,4,2)));
            dV[loc,loc,loc,loc] += expv;
        end    
        
        # ---
        @sync begin

            for i in 1:loc-1
                @Threads.spawn begin
                    dK[loc,i] += fast_expval(indmap_1L[2,i],r_end,bp_ut)
                    dK[i,loc] += fast_expval(indmap_1L[1,i],r_end,bm_ut)

                    expv = fast_expval(indmap_1L[2,i],r_end,ppLm);
                    dV[loc,loc,i,loc] += expv;
                    dV[loc,loc,loc,i] += expv;

                    expv = fast_expval(indmap_1L[1,i],r_end,Lpmm);
                    dV[loc,i,loc,loc] += expv;
                    dV[i,loc,loc,loc] += expv;
                end

            end

            # ---

            for j in loc+1:basis_size      
                @Threads.spawn begin      
                    expv = fast_expval(1,indmap_1R[2,j],ppRm);
                    dV[loc,loc,j,loc] += expv;
                    dV[loc,loc,loc,j] += expv;

                    expv = fast_expval(1,indmap_1R[1,j],Rpmm);
                    dV[j,loc,loc,loc] += expv;
                    dV[loc,j,loc,loc] += expv;
                end
            end
            
            # ---

            # 1 2 1
            for i in 1:half_basis_size,j in i+1:half_basis_size-1
                j == loc || continue;

            
                for k in j+1:basis_size
                    @Threads.spawn begin
                        expv = fast_expval(indmap_1L[1,i],indmap_1R[1,k],LRmm)
                        dV[k,i,j,j] += expv;
                        dV[i,k,j,j] += expv

                        expv = fast_expval(indmap_1L[2,i],indmap_1R[2,k],ppLR);
                        dV[j,j,k,i] += expv;
                        dV[j,j,i,k] += expv;

                        expv = fast_expval(indmap_1L[1,i],indmap_1R[2,k],LpRm);
                        dV[j,i,j,k] += expv;
                        dV[i,j,k,j] += expv;

                        expv = fast_expval(indmap_1L[1,i],indmap_1R[2,k],p_pm_p);
                        dV[j,i,k,j] += expv;
                        dV[i,j,j,k] += expv;

                        expv = fast_expval(indmap_1L[2,i],indmap_1R[1,k],RpLm);
                        dV[j,k,j,i] += expv;
                        dV[k,j,i,j] += expv;

                        expv = fast_expval(indmap_1L[2,i],indmap_1R[1,k],m_pm_m);
                        dV[j,k,i,j] += expv;
                        dV[k,j,j,i] += expv;
                    end
                end

            end

            # 2 1 1
            for i in 1:half_basis_size,j in i+1:half_basis_size
                j == loc || continue;

                for k in j+1:basis_size

                    @Threads.spawn begin
                        expv = fast_expval(indmap_2L[1,i,1,i],indmap_1R[2,k],bm_m);
                        dV[i,i,j,k] += expv;
                        dV[i,i,k,j] += expv
                        
                        expv = fast_expval(indmap_2L[2,i,1,i],indmap_1R[2,k],LpLR_1);
                        dV[j,i,i,k] -= 2*expv;
                        dV[i,j,k,i] -= 2*expv;

                        expv = fast_expval(indmap_2L[2,i,1,i],indmap_1R[2,k],bp_m);
                        dV[i,j,i,k] += expv;
                        dV[j,i,k,i] += expv;

                        expv = fast_expval(indmap_2L[2,i,1,i],indmap_1R[1,k],LRLm_1);
                        dV[i,k,i,j] += 2*expv;
                        dV[k,i,j,i] += 2*expv;
                        dV[i,k,j,i] -= 2*expv;
                        dV[k,i,i,j] -= 2*expv;
                        
                        expv = fast_expval(indmap_2L[2,i,1,i],indmap_1R[1,k],bm_p);
                        dV[i,k,i,j] -= expv;
                        dV[k,i,j,i] -= expv;

                        expv = fast_expval(indmap_2L[2,i,2,i],indmap_1R[1,k],RpLL);
                        dV[j,k,i,i] += expv;
                        dV[k,j,i,i] += expv;
                    end
                end
            end


            # 1 2 1
            for j in half_basis_size:basis_size, k in j+1:basis_size
                j == loc || continue;

                for i in 1:j-1
                    @Threads.spawn begin
                        expv = fast_expval(indmap_1L[1,i],indmap_1R[1,k],LRmm);
                        dV[k,i,j,j]+=expv;
                        dV[i,k,j,j]+=expv;

                        expv = fast_expval(indmap_1L[2,i],indmap_1R[2,k],ppLR);
                        dV[j,j,k,i]+=expv;
                        dV[j,j,i,k]+=expv;

                        expv = fast_expval(indmap_1L[1,i],indmap_1R[2,k],LpRm);
                        dV[j,i,j,k]+=expv;
                        dV[i,j,k,j]+=expv;

                        expv = fast_expval(indmap_1L[1,i],indmap_1R[2,k],p_pm_p);
                        dV[j,i,k,j]+=expv;
                        dV[i,j,j,k]+=expv;

                        expv = fast_expval(indmap_1L[2,i],indmap_1R[1,k],RpLm);
                        dV[j,k,j,i]+=expv;
                        dV[k,j,i,j]+=expv;

                        expv = fast_expval(indmap_1L[2,i],indmap_1R[1,k],m_pm_m);
                        dV[j,k,i,j]+=expv;
                        dV[k,j,j,i]+=expv;
                    end

                end
            end 

            # 1 1 2
            for j in half_basis_size:basis_size, k in j+1:basis_size
                j == loc || continue;

                for i in 1:j-1
                    @Threads.spawn begin
                        expv = fast_expval(indmap_1L[1,i],indmap_2R[2,k,2,k],jimm)
                        dV[j,i,k,k] +=expv;
                        dV[i,j,k,k] +=expv;

                        expv = fast_expval(indmap_1L[2,i],indmap_2R[1,k,1,k],ppji);
                        dV[k,k,j,i]+=expv;
                        dV[k,k,i,j]+=expv;

                        expv = fast_expval(indmap_1L[2,i],indmap_2R[2,k,1,k],jpim_1);
                        dV[j,k,i,k]+=expv;
                        dV[k,j,k,i]+=expv;
                        dV[k,j,i,k]-=2*expv;
                        dV[j,k,k,i]-=2*expv;

                        expv = fast_expval(indmap_1L[2,i],indmap_2R[2,k,1,k],jpim_1-m_ap);
                        dV[j,k,i,k]+=expv;
                        dV[k,j,k,i]+=expv;

                        expv = fast_expval(indmap_1L[1,i],indmap_2R[2,k,1,k],ipjm_1);
                        dV[i,k,j,k]+=expv;
                        dV[k,i,k,j]+=expv;
                        dV[i,k,k,j]-=2*expv;
                        dV[k,i,j,k]-=2*expv;
                        
                        expv = fast_expval(indmap_1L[1,i],indmap_2R[2,k,1,k],p_am - ipjm_1);
                        dV[i,k,j,k]+=expv;
                        dV[k,i,k,j]+=expv;
                    end
                end

            end
            
            

            # 1 1 2 + 2 2
            for k in 2:half_basis_size
                k == loc || continue;

                for i in 1:min(half_basis_size-1,k-1)
                    @Threads.spawn begin
                        dV[i,i,k,k] += fast_expval(indmap_2L[1,i,1,i],r_end,bmbm_ut);
                        dV[k,k,i,i] += fast_expval(indmap_2L[2,i,2,i],r_end,bpbp_ut);
                        
                        expv = fast_expval(indmap_2L[2,i,1,i],r_end,bpbm_ut);
                        dV[k,i,k,i]+=expv;
                        dV[i,k,i,k]+=expv;

                        expv = fast_expval(indmap_2L[2,i,1,i],r_end,_pm_left);
                        dV[k,i,i,k]+=expv;
                        dV[i,k,k,i]+=expv;
                    end     
                end

                for i in 1:k-1,j in i+1:k-1
                    @Threads.spawn begin
                        expv = fast_expval(indmap_2L[1,i,1,j],r_end,bmbm_ut);
                        dV[i,j,k,k]+=expv;
                        dV[j,i,k,k]+=expv;

                        expv = fast_expval(indmap_2L[2,i,2,j],r_end,bpbp_ut);
                        dV[k,k,j,i]+=expv;
                        dV[k,k,i,j]+=expv;

                        expv = fast_expval(indmap_2L[2,i,1,j],r_end,_pm_left);
                        dV[k,j,k,i]-=expv;
                        dV[j,k,i,k]-=expv;
                        dV[j,k,k,i]+=expv;
                        dV[k,j,i,k]+=expv;

                        expv = fast_expval(indmap_2L[1,i,2,j],r_end,_pm_left);
                        dV[i,k,k,j]+=expv;
                        dV[k,i,j,k]+=expv;

                        expv = fast_expval(indmap_2L[1,i,2,j],r_end,bmbp_ut);
                        dV[k,i,k,j]+=expv;
                        dV[i,k,j,k]+=expv;

                        expv = fast_expval(indmap_2L[2,i,1,j],r_end,bmbp_ut);
                        dV[k,j,k,i]-=expv;
                        dV[j,k,i,k]-=expv;
                    end
                end
            end
            
            
            # 2 1 1 + 2 2
            for i in half_basis_size:basis_size
                i == loc || continue;

                for j in i+1:basis_size
                    @Threads.spawn begin
                        dV[j,j,i,i] += fast_expval(1,indmap_2R[1,j,1,j],ut_amam)
                        dV[i,i,j,j] += fast_expval(1,indmap_2R[2,j,2,j],ut_apap);

                        expv = fast_expval(1,indmap_2R[2,j,1,j],ut_apam);
                        dV[i,j,i,j]+=expv;
                        dV[j,i,j,i]+=expv;

                        expv = fast_expval(1,indmap_2R[2,j,1,j],_pm_right);
                        dV[j,i,i,j] += expv;
                        dV[i,j,j,i]+=expv;
                    end
                end
                for j in i+1:basis_size,k in j+1:basis_size
                    @Threads.spawn begin
                        expv = fast_expval(1,indmap_2R[1,j,1,k],ut_amam)
                        dV[j,k,i,i] += expv;
                        dV[k,j,i,i] += expv;

                        expv = fast_expval(1,indmap_2R[2,j,2,k],ut_apap);
                        dV[i,i,j,k] += expv;
                        dV[i,i,k,j] += expv;

                        expv = fast_expval(1,indmap_2R[2,j,1,k],_pm_right);
                        dV[i,k,i,j] -= expv;
                        dV[k,i,j,i] -= expv;
                        dV[k,i,i,j] += expv;
                        dV[i,k,j,i] += expv;

                        expv = fast_expval(1,indmap_2R[1,j,2,k],_pm_right);
                        dV[i,j,k,i] += expv;
                        dV[j,i,i,k] += expv;

                        expv = fast_expval(1,indmap_2R[1,j,2,k],ut_amap);
                        dV[j,i,k,i] += expv;
                        dV[i,j,i,k] += expv;

                        expv = fast_expval(1,indmap_2R[2,j,1,k],ut_amap);
                        dV[i,k,i,j] -= expv;
                        dV[k,i,j,i] -= expv;
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

                
                for i in 1:k-1, j in i+1:k-1
                    @Threads.spawn begin

                        expv = fast_expval(indmap_2L[2,i,1,j],indmap_1R[2,l],LpLR_1);

                        dV[j,k,l,i] -= 2*expv;
                        dV[k,j,i,l] -= 2*expv;

                        expv = fast_expval(indmap_2L[1,i,2,j],indmap_1R[2,l],LpLR_1);

                        dV[i,k,l,j] -= 2*expv;
                        dV[k,i,j,l] -= 2*expv;
                        dV[k,i,l,j] += 2*expv;
                        dV[i,k,j,l] += 2*expv

                        
                        expv = fast_expval(indmap_2L[2,i,1,j],indmap_1R[2,l],bp_m);
                        dV[k,j,l,i] += expv;
                        dV[j,k,i,l] += expv;


                        expv = fast_expval(indmap_2L[1,i,2,j],indmap_1R[2,l],bp_m);
                        dV[k,i,l,j] -= expv;
                        dV[i,k,j,l] -= expv;

                        expv = fast_expval(indmap_2L[2,i,1,j],indmap_1R[1,l],LRLm_1)
                        dV[j,l,k,i]-=2*expv;
                        dV[l,j,i,k]-=2*expv;
                        dV[l,j,k,i]+=2*expv;
                        dV[j,l,i,k]+=2*expv;
                        
                        expv = fast_expval(indmap_2L[1,i,2,j],indmap_1R[1,l],LRLm_1)
                        dV[i,l,k,j]-=2*expv;
                        dV[l,i,j,k]-=2*expv;


                        expv = fast_expval(indmap_2L[2,i,1,j],indmap_1R[1,l],bm_p)
                        dV[l,j,k,i] -= expv;
                        dV[j,l,i,k] -= expv;
                    
                        expv = fast_expval(indmap_2L[1,i,2,j],indmap_1R[1,l],bm_p)
                        dV[l,i,k,j] += expv;
                        dV[i,l,j,k] += expv;
                        
                        expv = fast_expval(indmap_2L[1,i,1,j],indmap_1R[2,l],jimR_1)
                        dV[j,i,l,k] += 2*expv;
                        dV[i,j,k,l] += 2*expv;


                        expv = fast_expval(indmap_2L[1,i,1,j],indmap_1R[2,l],bm_m)
                        dV[i,j,l,k] += expv;
                        dV[j,i,k,l] += expv;
                        dV[j,i,l,k] -= expv;
                        dV[i,j,k,l] -= expv;

                        expv = fast_expval(indmap_2L[2,i,2,j],indmap_1R[1,l],RpLL)
                        dV[l,k,j,i] += expv;
                        dV[k,l,i,j] += expv;

                        expv = fast_expval(indmap_2L[2,i,2,j],indmap_1R[1,l],bp_p)
                        dV[l,k,i,j] += expv;
                        dV[k,l,j,i] += expv;
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

                for l in k+1:basis_size
                    @Threads.spawn begin

                        expv = fast_expval(indmap_2L[2,i,1,j],indmap_1R[2,l],LpLR_1);
                        dV[j,k,l,i]-=2*expv;
                        dV[k,j,i,l]-=2*expv;
                    
                        expv = fast_expval(indmap_2L[2,i,1,j],indmap_1R[2,l],bp_m);
                        dV[k,j,l,i]+=expv;
                        dV[j,k,i,l]+=expv;

                        expv = fast_expval(indmap_2L[1,i,2,j],indmap_1R[2,l],bp_m);
                        dV[k,i,l,j]-=expv;
                        dV[i,k,j,l]-=expv;

                        expv = fast_expval(indmap_2L[1,i,2,j],indmap_1R[2,l],LpLR_1);
                        dV[i,k,l,j]-=2*expv;
                        dV[k,i,j,l]-=2*expv;
                        dV[k,i,l,j]+=2*expv;
                        dV[i,k,j,l]+=2*expv;

                        expv = fast_expval(indmap_2L[2,i,1,j],indmap_1R[1,l],LRLm_1);
                        dV[j,l,k,i]-=2*expv;
                        dV[l,j,i,k]-=2*expv;
                        dV[l,j,k,i]+=2*expv;
                        dV[j,l,i,k]+=2*expv;

                        expv = fast_expval(indmap_2L[2,i,1,j],indmap_1R[1,l],bm_p);
                        dV[l,j,k,i]-=expv;
                        dV[j,l,i,k]-=expv;

                        expv = fast_expval(indmap_2L[1,i,2,j],indmap_1R[1,l],LRLm_1);
                        dV[i,l,k,j]-=2*expv;
                        dV[l,i,j,k]-=2*expv;


                        expv = fast_expval(indmap_2L[1,i,2,j],indmap_1R[1,l],bm_p);
                        dV[l,i,k,j]+=expv;
                        dV[i,l,j,k]+=expv;

                        expv = fast_expval(indmap_2L[1,i,1,j],indmap_1R[2,l],jimR_1);
                        dV[j,i,l,k]+=2*expv;
                        dV[i,j,k,l]+=2*expv;

                        expv = fast_expval(indmap_2L[1,i,1,j],indmap_1R[2,l],bm_m);
                        dV[i,j,l,k]+=expv;
                        dV[j,i,k,l]+=expv;
                        dV[j,i,l,k]-=expv;
                        dV[i,j,k,l]-=expv;

                        expv = fast_expval(indmap_2L[2,i,2,j],indmap_1R[1,l],RpLL);
                        dV[l,k,j,i]+=expv;
                        dV[k,l,i,j]+=expv;


                        expv = fast_expval(indmap_2L[2,i,2,j],indmap_1R[1,l],bp_p);
                        dV[l,k,i,j]+=expv;
                        dV[k,l,j,i]+=expv;
                    end
                end
            end

            # 3 right of half_basis_size
            for i in 1:basis_size,j in i+1:basis_size
                j >= half_basis_size || continue
                
                numblocks = 0
                for a in max(j+1,half_basis_size+1):basis_size, b in a+1:basis_size
                    numblocks += 1
                end
                numblocks/8 > (j-1)/12 || continue
                numblocks == 0 && continue

                j == loc || continue;

                for k in max(j+1,half_basis_size+1):basis_size,l in k+1:basis_size
                    @Threads.spawn begin

                        expv = fast_expval(indmap_1L[2,i],indmap_2R[1,k,2,l],-2*jpim_1);
                        dV[j,k,i,l]-=expv/2
                        dV[k,j,l,i]-=expv/2
                        dV[k,j,i,l]+=expv;
                        dV[j,k,l,i]+=expv;

                        expv = fast_expval(indmap_1L[2,i],indmap_2R[2,k,1,l],-2*jpim_1);
                        dV[l,j,k,i]-=expv/2
                        dV[j,l,i,k]-=expv/2
                        dV[l,j,i,k]+=expv;
                        dV[j,l,k,i]+=expv;

                        expv = fast_expval(indmap_1L[2,i],indmap_2R[1,k,2,l],m_ap-jpim_1);
                        dV[j,k,i,l]+=expv;
                        dV[k,j,l,i]+=expv;
                        
                        expv = fast_expval(indmap_1L[2,i],indmap_2R[2,k,1,l],m_ap-jpim_1);
                        dV[l,j,k,i]-=expv;
                        dV[j,l,i,k]-=expv;

                        expv = fast_expval(indmap_1L[1,i],indmap_2R[1,k,2,l],-2*ipjm_1);
                        dV[i,k,j,l]-=expv/2
                        dV[k,i,l,j]-=expv/2
                        dV[k,i,j,l]+=expv;
                        dV[i,k,l,j]+=expv;

                        expv = fast_expval(indmap_1L[1,i],indmap_2R[2,k,1,l],-2*ipjm_1);
                        dV[l,i,k,j]-=expv/2
                        dV[i,l,j,k]-=expv/2
                        dV[l,i,j,k]+=expv;
                        dV[i,l,k,j]+=expv;

                        expv = fast_expval(indmap_1L[1,i],indmap_2R[1,k,2,l],-(p_am-ipjm_1));
                        dV[i,k,j,l]+=expv;
                        dV[k,i,l,j]+=expv;

                        expv = fast_expval(indmap_1L[1,i],indmap_2R[2,k,1,l],-(p_am-ipjm_1));
                        dV[l,i,k,j]-=expv;
                        dV[i,l,j,k]-=expv;

                        expv = fast_expval(indmap_1L[2,i],indmap_2R[1,k,1,l],(m_am + ppji)/2);
                        dV[l,k,j,i]+=expv;
                        dV[k,l,i,j]+=expv;
                        dV[l,k,i,j]+=expv;
                        dV[k,l,j,i]+=expv;

                        expv = fast_expval(indmap_1L[2,i],indmap_2R[1,k,1,l],(m_am - ppji)/2);
                        dV[l,k,j,i]-=expv;
                        dV[k,l,i,j]-=expv;
                        dV[l,k,i,j]+=expv;
                        dV[k,l,j,i]+=expv;

                        expv = fast_expval(indmap_1L[1,i],indmap_2R[2,k,2,l],(p_ap+jimm)/2);
                        dV[j,i,k,l]+=expv;
                        dV[i,j,l,k]+=expv;
                        dV[i,j,k,l]+=expv;
                        dV[j,i,l,k]+=expv;

                        expv = fast_expval(indmap_1L[1,i],indmap_2R[2,k,2,l],(p_ap-jimm)/2);
                        dV[j,i,k,l]+=expv;
                        dV[i,j,l,k]+=expv;
                        dV[i,j,k,l]-=expv;
                        dV[j,i,l,k]-=expv;
                    end

                end

            end
                        
            
            for j in half_basis_size:basis_size,k in max(j+1,half_basis_size+1):basis_size,l in k+1:basis_size
                j >= half_basis_size || continue
                
                numblocks = 0
                for a in max(j+1,half_basis_size+1):basis_size, b in a+1:basis_size
                    numblocks += 1
                end
                numblocks/8 <= (j-1)/12 || continue
                numblocks == 0 && continue
                
                j==loc || continue

                for i in 1:j-1

                    @Threads.spawn begin
                        expv = fast_expval(indmap_1L[2,i],indmap_2R[1,k,2,l],-2*jpim_1);
                        dV[j,k,i,l]-=expv/2
                        dV[k,j,l,i]-=expv/2
                        dV[k,j,i,l]+=expv;
                        dV[j,k,l,i]+=expv;

                        expv = fast_expval(indmap_1L[2,i],indmap_2R[2,k,1,l],-2*jpim_1);
                        dV[l,j,k,i]-=expv/2
                        dV[j,l,i,k]-=expv/2
                        dV[l,j,i,k]+=expv
                        dV[j,l,k,i]+=expv;

                        expv = fast_expval(indmap_1L[2,i],indmap_2R[1,k,2,l],m_ap-jpim_1);
                        dV[j,k,i,l]+=expv;
                        dV[k,j,l,i]+=expv;

                        expv = fast_expval(indmap_1L[2,i],indmap_2R[2,k,1,l],m_ap-jpim_1);
                        dV[l,j,k,i]-=expv;
                        dV[j,l,i,k]-=expv;

                        expv = fast_expval(indmap_1L[1,i],indmap_2R[1,k,2,l],-2*ipjm_1);
                        dV[i,k,j,l]-=expv/2
                        dV[k,i,l,j]-=expv/2
                        dV[k,i,j,l]+=expv;
                        dV[i,k,l,j]+=expv;

                        expv = fast_expval(indmap_1L[1,i],indmap_2R[2,k,1,l],-2*ipjm_1);
                        dV[l,i,k,j]-=expv/2
                        dV[i,l,j,k]-=expv/2
                        dV[l,i,j,k]+=expv;
                        dV[i,l,k,j]+=expv;

                        expv = fast_expval(indmap_1L[1,i],indmap_2R[1,k,2,l],-(p_am-ipjm_1));
                        dV[i,k,j,l]+=expv;
                        dV[k,i,l,j]+=expv;

                        expv = fast_expval(indmap_1L[1,i],indmap_2R[2,k,1,l],-(p_am-ipjm_1));
                        dV[l,i,k,j]-=expv;
                        dV[i,l,j,k]-=expv;

                        expv = fast_expval(indmap_1L[2,i],indmap_2R[1,k,1,l],(m_am + ppji)/2);
                        dV[l,k,j,i]+=expv;
                        dV[k,l,i,j]+=expv;
                        dV[l,k,i,j]+=expv;
                        dV[k,l,j,i]+=expv;

                        expv = fast_expval(indmap_1L[2,i],indmap_2R[1,k,1,l],(m_am - ppji)/2);
                        dV[l,k,j,i]-=expv;
                        dV[k,l,i,j]-=expv;
                        dV[l,k,i,j]+=expv;
                        dV[k,l,j,i]+=expv

                        expv = fast_expval(indmap_1L[1,i],indmap_2R[2,k,2,l],(p_ap+jimm)/2);
                        dV[j,i,k,l]+=expv;
                        dV[i,j,l,k]+=expv;
                        dV[i,j,k,l]+=expv;
                        dV[j,i,l,k]+=expv;

                        expv = fast_expval(indmap_1L[1,i],indmap_2R[2,k,2,l],(p_ap-jimm)/2);
                        dV[j,i,k,l]+=expv;
                        dV[i,j,l,k]+=expv;
                        dV[i,j,k,l]-=expv;
                        dV[j,i,l,k]-=expv;
                    end
                end
                
            end

             # loc == half_basis_size: 
            for k in half_basis_size+1:basis_size, l in k+1:basis_size
                loc == half_basis_size || continue;

                for i in 1:half_basis_size-1,j in i+1:half_basis_size-1
                    @Threads.spawn begin
                        expv = fast_expval(indmap_2L[2,i,1,j],indmap_2R[1,k,2,l],pm_ai_pm);
                        dV[j,k,i,l]+=expv;
                        dV[k,j,l,i]+=expv;
                        dV[k,j,i,l]-=2*expv;
                        dV[j,k,l,i]-=2*expv;

                        expv = fast_expval(indmap_2L[1,i,2,j],indmap_2R[1,k,2,l],pm_ai_pm);
                        dV[k,i,j,l]-=2*expv;
                        dV[i,k,l,j]-=2*expv;
                        dV[i,k,j,l]+=expv;
                        dV[k,i,l,j]+=expv;

                        expv = fast_expval(indmap_2L[1,i,2,j],indmap_2R[2,k,1,l],pm_ai_pm);
                        dV[l,i,k,j]+=expv;
                        dV[i,l,j,k]+=expv;
                        dV[l,i,j,k]-=2*expv;
                        dV[i,l,k,j]-=2*expv;

                        expv = fast_expval(indmap_2L[2,i,1,j],indmap_2R[2,k,1,l],pm_ai_pm);
                        dV[l,j,k,i]+=expv;
                        dV[j,l,i,k]+=expv;
                        dV[l,j,i,k]-=2*expv;
                        dV[j,l,k,i]-=2*expv;


                        expv = fast_expval(indmap_2L[1,i,1,j],indmap_2R[2,k,2,l],pp_ai_pp);
                        dV[j,i,k,l]+=expv;
                        dV[i,j,l,k]+=expv;
                        dV[i,j,k,l]-=expv;
                        dV[j,i,l,k]-=expv;

                        expv = fast_expval(indmap_2L[2,i,2,j],indmap_2R[1,k,1,l],mm_ai_mm);
                        dV[l,k,i,j]+=expv;
                        dV[k,l,j,i]+=expv;
                        dV[l,k,j,i]-=expv;
                        dV[k,l,i,j]-=expv;


                        expv = fast_expval(indmap_2L[2,i,1,j],indmap_2R[1,k,2,l],jkil_2);
                        dV[k,j,i,l]+=2*expv;
                        dV[j,k,l,i]+=2*expv;

                        expv = fast_expval(indmap_2L[1,i,2,j],indmap_2R[1,k,2,l],jkil_2);
                        dV[k,i,j,l]+=2*expv;
                        dV[i,k,l,j]+=2*expv;
                        dV[i,k,j,l]-=2*expv;
                        dV[k,i,l,j]-=2*expv;

                        expv = fast_expval(indmap_2L[1,i,2,j],indmap_2R[2,k,1,l],jkil_2);
                        dV[l,i,j,k]+=2*expv;
                        dV[i,l,k,j]+=2*expv

                        expv = fast_expval(indmap_2L[2,i,1,j],indmap_2R[2,k,1,l],jkil_2);
                        dV[l,j,k,i]-=2*expv;
                        dV[j,l,i,k]-=2*expv;
                        dV[l,j,i,k]+=2*expv;
                        dV[j,l,k,i]+=2*expv;

                        expv = fast_expval(indmap_2L[1,i,1,j],indmap_2R[2,k,2,l],jikl_1);
                        dV[i,j,k,l]+=2*expv;
                        dV[j,i,l,k]+=2*expv;

                        expv = fast_expval(indmap_2L[2,i,2,j],indmap_2R[1,k,1,l],lkij_1);
                        dV[l,k,j,i]+=2*expv;
                        dV[k,l,i,j]+=2*expv;
                    end

                end
            end
            
            for k in half_basis_size+1:basis_size
                loc == half_basis_size || continue;

                for i in 1:half_basis_size-1
                    @Threads.spawn begin
                        expv = fast_expval(indmap_2L[1,i,2,i],indmap_2R[2,k,1,k],pm_ai_pm);
                        dV[i,k,i,k]+=expv;
                        dV[k,i,k,i]+=expv

                        expv = fast_expval(indmap_2L[2,i,1,i],indmap_2R[2,k,1,k],pm_ai_pm);
                        dV[i,k,k,i]-=2*expv;
                        dV[k,i,i,k]-=2*expv;

                        expv = fast_expval(indmap_2L[2,i,1,i],indmap_2R[2,k,1,k],jkil_2);
                        dV[i,k,k,i]+=2*expv;
                        dV[k,i,i,k]+=2*expv;

                        expv = fast_expval(indmap_2L[2,i,2,i],indmap_2R[1,k,1,k],mm_ai_mm);
                        dV[k,k,i,i]+=expv;

                        expv = fast_expval(indmap_2L[1,i,1,i],indmap_2R[2,k,2,k],pp_ai_pp);
                        dV[i,i,k,k]+=expv;
                    end
                end

                for i in 1:half_basis_size-1, j in i+1:half_basis_size-1
                    @Threads.spawn begin

                        expv = fast_expval(indmap_2L[2,i,1,j],indmap_2R[2,k,1,k],pm_ai_pm);
                        dV[j,k,i,k]+=expv;
                        dV[k,j,k,i]+=expv;
                        dV[k,j,i,k]-=2*expv;
                        dV[j,k,k,i]-=2*expv;

                        expv = fast_expval(indmap_2L[1,i,2,j],indmap_2R[2,k,1,k],pm_ai_pm);
                        dV[i,k,j,k]+=expv;
                        dV[k,i,k,j]+=expv;
                        dV[i,k,k,j]-=2*expv;
                        dV[k,i,j,k]-=2*expv

                        expv = fast_expval(indmap_2L[2,i,1,j],indmap_2R[2,k,1,k],jkil_2);
                        dV[j,k,i,k]-=2*expv;
                        dV[k,j,k,i]-=2*expv;
                        dV[k,j,i,k]+=2*expv;
                        dV[j,k,k,i]+=2*expv;

                        expv = fast_expval(indmap_2L[1,i,2,j],indmap_2R[2,k,1,k],jkil_2);
                        dV[i,k,k,j]+=2*expv;
                        dV[k,i,j,k]+=2*expv

                        expv = fast_expval(indmap_2L[2,i,2,j],indmap_2R[1,k,1,k],mm_ai_mm);
                        dV[k,k,j,i]+=expv;
                        dV[k,k,i,j]+=expv;

                        expv = fast_expval(indmap_2L[1,i,1,j],indmap_2R[2,k,2,k],pp_ai_pp);
                        dV[j,i,k,k]+=expv;
                        dV[i,j,k,k]+=expv;
                    end
                end
                    
            end
            
            for i in 1:half_basis_size-1
                loc == half_basis_size || continue;

                for j in half_basis_size+1:basis_size,k in j+1:basis_size
                    @Threads.spawn begin

                        expv = fast_expval(indmap_2L[2,i,2,i],indmap_2R[1,j,1,k],mm_ai_mm);
                        dV[j,k,i,i]+=expv;
                        dV[k,j,i,i]+=expv;

                        expv = fast_expval(indmap_2L[1,i,1,i],indmap_2R[2,j,2,k],pp_ai_pp);
                        dV[i,i,j,k]+=expv;
                        dV[i,i,k,j]+=expv;

                        expv = fast_expval(indmap_2L[2,i,1,i],indmap_2R[1,j,2,k],pm_ai_pm);
                        dV[j,i,i,k]-=2*expv;
                        dV[i,j,k,i]-=2*expv;
                        dV[i,j,i,k]+=expv;
                        dV[j,i,k,i]+=expv;

                        expv = fast_expval(indmap_2L[2,i,1,i],indmap_2R[2,j,1,k],pm_ai_pm);
                        dV[k,i,i,j]-=2*expv;
                        dV[i,k,j,i]-=2*expv;
                        dV[k,i,j,i]+=expv;
                        dV[i,k,i,j]+=expv;

                        expv = fast_expval(indmap_2L[2,i,1,i],indmap_2R[1,j,2,k],jkil_2);
                        dV[j,i,i,k]+=2*expv;
                        dV[i,j,k,i]+=2*expv;

                        expv = fast_expval(indmap_2L[2,i,1,i],indmap_2R[2,j,1,k],jkil_2);
                        dV[k,i,i,j]+=2*expv;
                        dV[i,k,j,i]+=2*expv;
                        dV[k,i,j,i]-=2*expv;
                        dV[i,k,i,j]-=2*expv

                    end
                end

            end
        
        end
        
        
       
    end

    energy = real(sum(expectation_value(state, qchemham, envs)))
    #@show energy
    (dV,dK,dAC,energy)
end

function project_U(odV,odK,ham::CASSCF_Ham)
    h = transform(ham);
    
    # 1) dV and dK are calculated over the active space, they should be extended to include the full space
    dV = zero.(ham.ERI);
    dK = zero.(ham.K);
    dV[ham.active,ham.active,ham.active,ham.active] .= odV;
    dK[ham.active,ham.active] .= odK;
    closed = 1:(first(ham.active)-1);


    for c in closed
        dK[c,c] = 2;

        dV[c,c,c,c] += 2
        for n in closed
            if n != c
                dV[n,c,c,n] += 4;
                #dV[c,n,n,c] += 4;
                dV[c,n,c,n] -= 2;
                #dV[n,c,n,c] -= 2;
            end
        end

        for n in ham.active,m in ham.active
            dV[n,c,c,m] += dK[n,m]*2;
            dV[c,n,m,c] += dK[n,m]*2;
            dV[c,n,c,m] -= dK[n,m];
            dV[n,c,m,c] -= dK[n,m];
        end
    end
    
    (K,ERI) = transform(ham)

    #@show ham.Vnuc+sum(K.*dK)+sum(ERI.*dV)

    @tensor F[-1;-2] := -K[1;-2]*dK[1;-1]+K[-1;1]*dK[-2;1]
    
    @tensor F[-1;-2] -= ERI[1 2;-2 4]*dV[1 2;-1 4]
    @tensor F[-1;-2] -= ERI[1 2;4 -2]*dV[1 2;4 -1]
    
    @tensor F[-1;-2] += ERI[-1 2;3 4]*dV[-2 2;3 4]
    @tensor F[-1;-2] += ERI[2 -1;3 4]*dV[2 -2;3 4]
    
    #=
    @tensor L[-1;-2] := 0.5*K[-1;4]*dK[-2;4]
    @tensor L[-1;-2] += 0.5*K[4;-2]*dK[4;-1]
    @tensor L[-1;-2] += 0.5*ERI[-1 4;2 3]*dV[-2 4;2 3]
    @tensor L[-1;-2] += 0.5*ERI[4 -1;2 3]*dV[4 -2;2 3]
    @tensor L[-1;-2] += 0.5*ERI[4 2;3 -2]*dV[4 2;3 -1]
    @tensor L[-1;-2] += 0.5*ERI[4 2;-2 3]*dV[4 2;-1 3]
    =#

    @tensor R[-1;-2] := 0.5*K[-1;2]*dK[-2;2]
    @tensor R[-1;-2] += 0.5*K[2;-2]*dK[2;-1]
    @tensor R[-1;-2] += 0.5*ERI[-1 1;2 3]*dV[-2 1;2 3]
    @tensor R[-1;-2] += 0.5*ERI[1 -1;2 3]*dV[1 -2;2 3]
    @tensor R[-1;-2] += 0.5*ERI[1 2;3 -2]*dV[1 2;3 -1]
    @tensor R[-1;-2] += 0.5*ERI[1 2;-2 3]*dV[1 2;-1 3]
    
    function apply_hessian(v)
        
        
        
        t = -v*R
        #@tensor y[-1;-2] := -v[-1;1]*R[1;-2]
        #@tensor y[-1;-2] -= L[-1;1]*v[1;-2]
        
        
        @tensor t[-1;-2] += 2*ERI[-1 2;1 3]*dV[-2 2;4 3]*v[1;4]
        #@tensor y[-1;-2] += 2*ERI[2 4;3 -2]*dV[2 1;3 -1]*v[1;4]

        @tensor t[-1;-2] += 2*ERI[-1 2;3 1]*dV[-2 2;3 4]*v[1;4]
        #@tensor y[-1;-2] += 2*ERI[4 2;3 -2]*dV[1 2;3 -1]*v[1;4]


        @tensor t[-1;-2] -= 2*ERI[-1 4;2 3]*dV[-2 1;2 3]*v[1;4]
        #@tensor y[-1;-2] -= 2*ERI[3 2;1 -2]*dV[3 2;4 -1]*v[1;4]    

        y = t-t'

        @tensor y[-1;-2] += K[-1;1]*v[1;2]*dK[-2;2]
        @tensor y[-1;-2] += K[4;-2]*dK[2;-1]*v[2;4]


        y        
    end
    
    #=
    ou = one(K);
    hessian = ERI.*0;
    @tensor hessian[-1 -2;-3 -4] := -0.5*ou[-1;-3]*K[-4;1]*dK[-2;1]
    @tensor hessian[-1 -2;-3 -4] -= 0.5*K[1;-2]*dK[1;-4]*ou[-1;-3]
    

    @tensor hessian[-1 -2;-3 -4] += K[-1;-3]*dK[-2;-4]
    
    @tensor hessian[-1 -2;-3 -4] += ERI[-1 1;-3 3]*dV[-2 1;-4 3]
    @tensor hessian[-1 -2;-3 -4] += ERI[1 -1;-3 3]*dV[1 -2;-4 3]
    
    @tensor hessian[-1 -2;-3 -4] += ERI[-1 1;3 -3]*dV[-2 1;3 -4]
    @tensor hessian[-1 -2;-3 -4] += ERI[1 -1;3 -3]*dV[1 -2;3 -4]

    @tensor hessian[-1 -2;-3 -4] -= ERI[-1 -4;2 3]*dV[-2 -3;2 3]
    @tensor hessian[-1 -2;-3 -4] -= ERI[2 3;-2 -3]*dV[2 3;-1 -4]    

    @tensor hessian[-1 -2;-3 -4] -= 0.5*ou[-1;-3]*ERI[-4 1;2 3]*dV[-2 1;2 3]
    @tensor hessian[-1 -2;-3 -4] -= 0.5*ou[-1;-3]*ERI[1 -4;2 3]*dV[1 -2;2 3]

    @tensor hessian[-1 -2;-3 -4] -= 0.5*ERI[1 2;3 -2]*dV[1 2;3 -4]*ou[-1;-3]
    @tensor hessian[-1 -2;-3 -4] -= 0.5*ERI[1 2;-2 3]*dV[1 2;-4 3]*ou[-1;-3]
    hessian += permutedims(hessian,(4,3,2,1))
    =#
    g = -F
    
    g,apply_hessian
end


function transform(h::CASSCF_Ham)
    @tensor c_eri[-1 -2;-3 -4] := h.U[-1;1]*h.U[-2;2]*conj(h.U[-3;3])*conj(h.U[-4;4])*h.ERI[1 2;3 4]
    c_k = h.U*h.K*(h.U')
    return (c_k,c_eri)
end

function mpo_representation(h::CASSCF_Ham)
    (K,ERI) = transform(h);

    E = h.Vnuc;

    t_ERI = copy(ERI[h.active,h.active,h.active,h.active]);
    t_K = copy(K[h.active,h.active]);

    # ERI/K need to be modified to include mean field filled orbitals

    filled = 1:(h.active.start-1);
    

    for a in filled, b in filled
        if a == b
            E += ERI[a,b,b,a]*2
            E += K[a,a]*2
        else
            E += 4*ERI[a,b,b,a]
            E -= 2*ERI[a,b,a,b]
        end
    end
    for a in filled
        t_K += ERI[h.active,a,a,h.active]*2
        t_K += ERI[a,h.active,h.active,a]*2

        t_K -= ERI[a,h.active,a,h.active]
        t_K -= ERI[h.active,a,h.active,a]
    end
    fused_quantum_chemistry_hamiltonian(E,t_K,t_ERI)
end


function manifoldpoint(state,ham)
    (dV,dK,dAC,E) = quantum_chemistry_dV_dK(ham,state);
    (dU,hessian) = project_U(dV,dK,ham);

    g = (Grassmann.project.(dAC,state.AL))

    Rhoreg = Vector{eltype(state.CR)}(undef,length(state));
    for i in 1:length(state)
        Rhoreg[i] = GrassmannMPS.regularize(state.CR[i],norm(g[i])/10)
    end

    (state,ham,Rhoreg,g,dU,hessian,E);
end

function cfun(x)
    (state,ham,Rhoreg,g,dU,hessian,E) = x;
    
    
    g_prec = map(1:length(state)) do i
        GrassmannMPS.PrecGrad(rmul!(copy(g[i]),state.CR[i]'),Rhoreg[i])
    end
    
    (E,(g_prec,dU))
end

function retract(x,g,alpha)
    flush(stderr); flush(stdout);
    (state,ham) = x;
    (g_prec,grad_U) = g;
    
    h = similar(g_prec);
    y = copy(state);

    for i in 1:length(state)
        (tal, th) = Grassmann.retract(state.AL[i], g_prec[i].Pg, alpha)
        
        h[i] = GrassmannMPS.PrecGrad(th);
        
        y.AC[i] = (tal,state.CR[i])
    end
    new_U = exp(alpha*grad_U)*ham.U

    newpoint = manifoldpoint(y,CASSCF_Ham(ham.ERI,ham.K,ham.Vnuc,new_U,ham.active));
    newgrad = (h,grad_U)
    (newpoint,newgrad)
end

function transport!(h, x, g, alpha, xp)
    (s_state,s_ham) = x;
    (p_state,p_ham) = xp;
    (h_state,h_ham) = h;
    (g_state,g_ham) = g;

    for i in 1:length(s_state)
        h_state[i] = GrassmannMPS.PrecGrad(Grassmann.transport!(h_state[i].Pg, s_state.AL[i], g_state[i].Pg, alpha, p_state.AL[i]))
    end

    
    E = exp((alpha/2)*g_ham)
    h_ham = E*h_ham*E'

    #h_ham = Unitary.transport!(h_ham,s_ham.U,g_ham,alpha,p_ham.U);
    return (h_state,h_ham)
end

function inner(x, g1, g2)
    (state,ham,rhoReg) = x;
    (g1_state,g1_ham) = g1;
    (g2_state,g2_ham) = g2;
    2 * real(sum(map(zip(rhoReg,g1_state,g2_state)) do (a,b,c)
        GrassmannMPS.inner(b,c,a)
    end))+real(dot(g1_ham,g2_ham))
end

function scale!(g, alpha)
    (g_state,g_ham) = g;
    (g_state .* alpha, g_ham*alpha)
end

function add!(g1, g2, alpha)
    (g1_state,g1_ham) = g1;
    (g2_state,g2_ham) = g2;

    (g1_state+g2_state*alpha,g1_ham+g2_ham*alpha)
end

function precondition(x,v)
    (g_prec,grad_U) = v;
    (state,ham,Rhoreg,g,dU,hessian,E) = x;
    
    shift_magn = norm(grad_U);

    # invert (Hessian + shift)
    # if it's a descent direction => continue
    # if not, increase shift
    shift = 0;
    while true
        (nsol,convhist) = linsolve(grad_U,grad_U,GMRES(maxiter=1,tol=1e-12)) do y
            t = hessian(y)
            #@tensor t[-1;-2] := hessian[-1,-2,1,2]*y[1,2]
            t += shift*y
        end
        #@show convhist.normres
        nsol = (nsol-nsol')/2
        
        if real(dot(dU,nsol)) >0
            return (g_prec,nsol)
        end
        @warn "not a descent direction, increasing shift $(dot(dU,nsol)) $(shift)+$(shift_magn)"
        shift +=shift_magn;
    end    
end

