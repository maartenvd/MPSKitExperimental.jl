#=
The chemistry hamiltonian is often treated by freezing a set of orbitals to be frozen-filled, another set to be frozen-empty, and the rest using an MPS
This structure represents this hamiltonian (the full ERI,K,Vnuc, along with the set of orbitals that are actively treated)
=#

struct GrassmannSCF <: MPSKit.Algorithm
    maxiter::Int
    tol::Float64
end

GrassmannSCF(;tol = 1e-12,maxiter=100) = GrassmannSCF(maxiter,tol);



struct CASSCF_Ham{B<:BasisSet,D,E}
    basis::B
    ao2mo::D
    active::E
end

function CASSCF_Ham(basis::BasisSet,active)
    C = overlap(basis);
    
    (vals,vecs) = eigen(C);
    
    CASSCF_Ham(basis,complex(vecs*diagm(sqrt.(vals).^-1)),active)
end

function MPSKit.find_groundstate(state,ham::CASSCF_Ham,alg::GrassmannSCF)

    (out,fx,_,_,normgradhistory) = optimize(cfun,manifoldpoint(state,ham), ConjugateGradient(gradtol=alg.tol, maxiter=alg.maxiter, verbosity = 2);
                retract = retract, inner = inner, transport! = transport! ,
                scale! = scale! , add! = add! , isometrictransport = true,precondition = precondition);
    st = out[1]; ham = out[2];
    return (st,ham)
    
    #=
    optimtest(cfun,manifoldpoint(state,ham);alpha = 0:0.01:0.1,
                retract = retract, inner = inner);
    =#
end

function _make_dVdK_factories(A)
    psp = space(A,2); #Vect[(Irrep[U₁]⊠Irrep[SU₂] ⊠ FermionParity)]((0,0,0)=>1, (1,1//2,1)=>1, (2,0,0)=>1);
    possible_mpo_spaces = [
        Vect[(Irrep[U₁]⊠Irrep[SU₂] ⊠ FermionParity)]((-1,1//2,1)=>1),
        Vect[(Irrep[U₁]⊠Irrep[SU₂] ⊠ FermionParity)]((1,1//2,1)=>1),
        Vect[(Irrep[U₁]⊠Irrep[SU₂] ⊠ FermionParity)]((2,1,0)=>1, (2,0,0)=>1),
        Vect[(Irrep[U₁]⊠Irrep[SU₂] ⊠ FermionParity)]((-2,1,0)=>1, (-2,0,0)=>1),
        Vect[(Irrep[U₁]⊠Irrep[SU₂] ⊠ FermionParity)]((0,0,0)=>1),
        Vect[(Irrep[U₁]⊠Irrep[SU₂] ⊠ FermionParity)]((0,0,0)=>1,(0,1,0)=>1)];
        
        #Vect[(Irrep[U₁] ⊠ Irrep[SU₂] ⊠ FermionParity)]((0, 0, 0)=>1, (2, 0, 0)=>1, (1, 1/2, 1)=>1)

    S = spacetype(A);
    storage = storagetype(A);
    
    tvaltype_1_2 = TransposeFactType(S,storage,1,2);
    tvaltype_2_2 = TransposeFactType(S,storage,2,2);
    mvaltype_2_2 = DelayedFactType(S,storage,2,2);
    mvaltype_2_1 = DelayedFactType(S,storage,2,1);
    tvaltype_3_0 = TransposeFactType(S,storage,3,0)
    tvaltype_0_3 = TransposeFactType(S,storage,0,3)
    
    tfactory_2_2 = Dict{Any,tvaltype_2_2}(); 
    mfactory_2_2 = Dict{Any,mvaltype_2_2}();
    mfactory_2_1 = Dict{Any,mvaltype_2_1}();
    tfactory_1_2 = Dict{Any,tvaltype_1_2}();
    tfactory_3_0 = Dict{Any,tvaltype_3_0}();
    tfactory_0_3 = Dict{Any,tvaltype_0_3}();
    
    
    v_1 = space(A,1);
    v_3 = space(A,3)';
    p = space(A,2);

    promise_creation = Dict{Any,Any}();

    for v in (v_1,v_3), mpo_virt_1 in vcat(possible_mpo_spaces,conj.(possible_mpo_spaces))
        temp_6_homsp = v*mpo_virt_1←v;
        if !(temp_6_homsp in keys(promise_creation))
            promise_creation[temp_6_homsp] = @Threads.spawn (mfactory_2_1,DelayedFact(temp_6_homsp,storage))
        end
        mothertask = promise_creation[temp_6_homsp]
        if !((temp_6_homsp,(),(3,2,1)) in keys(promise_creation))
            promise_creation[(temp_6_homsp,(),(3,2,1))] = @Threads.spawn (tfactory_0_3,TransposeFact(fetch(mothertask)[2],(),(3,2,1)));
        end

        if !((temp_6_homsp,(1,2,3),()) in keys(promise_creation))
            promise_creation[(temp_6_homsp,(1,2,3),())] = @Threads.spawn (tfactory_3_0,TransposeFact(fetch(mothertask)[2],(1,2,3),()));
        end
    end
    for mpo_virt_1 in possible_mpo_spaces, mpo_virt_4 in possible_mpo_spaces
        temp0_homsp = v_1*mpo_virt_1'←v_1;
        if !((temp0_homsp,(1,),(3,2)) in keys(promise_creation))
            promise_creation[(temp0_homsp,(1,),(3,2))] = @Threads.spawn (tfactory_1_2,TransposeFact(temp0_homsp,storage,(1,),(3,2)));
        end

        temp1_homsp = v_3*p'←v_1*mpo_virt_1;
        temp2_homsp = temp1_homsp;        
        if !((temp2_homsp,(3,1),(4,2)) in keys(promise_creation))
            t_t_t = @Threads.spawn (mfactory_2_2,DelayedFact(temp1_homsp,storage))
            promise_creation[temp1_homsp] = t_t_t
            promise_creation[(temp2_homsp,(3,1),(4,2))] = @Threads.spawn (tfactory_2_2,TransposeFact(fetch(t_t_t)[2],(3,1),(4,2)))
        end

        temp3_homsp = v_1'*v_3←p*mpo_virt_4
        temp4_homsp = temp3_homsp;
        if !((temp4_homsp,(2,4),(1,3)) in keys(promise_creation))
            t_t =  @Threads.spawn (mfactory_2_2,DelayedFact(temp3_homsp,storage))
            promise_creation[temp3_homsp] = t_t
            promise_creation[(temp4_homsp,(2,4),(1,3))] = @Threads.spawn (tfactory_2_2,TransposeFact(fetch(t_t)[2],(2,4),(1,3)))
        end

        temp_5_homsp = v_3*mpo_virt_4'←v_3;
        if !(temp_5_homsp in keys(promise_creation))
            promise_creation[temp_5_homsp] = @Threads.spawn (mfactory_2_1,DelayedFact(temp_5_homsp,storage));
        end

        temp_1_homsp = v_3*mpo_virt_4 ← v_3;
        if !((temp_1_homsp,(1,),(3,2)) in keys(promise_creation))
            promise_creation[(temp_1_homsp,(1,),(3,2))] = @Threads.spawn (tfactory_1_2,TransposeFact(temp_1_homsp,storage,(1,),(3,2)))
        end

        temp_2_homsp = v_1*p ← v_3*mpo_virt_4';
        if !((temp_2_homsp,(2,4),(1,3)) in keys(promise_creation))
            task_2 = @Threads.spawn (mfactory_2_2,DelayedFact(temp_2_homsp,storage));
            promise_creation[temp_2_homsp] = task_2
            promise_creation[(temp_2_homsp,(2,4),(1,3))] = @Threads.spawn (tfactory_2_2,TransposeFact(fetch(task_2)[2],(2,4),(1,3)))
        end
        

        temp_4_homsp = mpo_virt_1*p←v_1'*v_3
        if !((temp_4_homsp,(3,1),(4,2)) in keys(promise_creation))
            task_4 = @Threads.spawn (mfactory_2_2,DelayedFact(temp_4_homsp,storage))
            promise_creation[temp_4_homsp] = task_4
            promise_creation[(temp_4_homsp,(3,1),(4,2))] = @Threads.spawn (tfactory_2_2,TransposeFact(fetch(task_4)[2],(3,1),(4,2)));
        end
        
    end
    
    for (k_t,t_t) in promise_creation
        #@show k_t
        (d_t,v_t) = fetch(t_t)
        Base.setindex!(d_t,v_t,k_t);
    end

    return (tfactory_2_2, mfactory_2_2, mfactory_2_1,tfactory_1_2, tfactory_0_3, tfactory_3_0);
    
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
    @plansor _pm_right[-1 -2; -3 -4] := ut[-1]*h_pm[-2;-3]*(transpose(Rmap_bpbm_to_pm*pm_f',(1,)))[-4]
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

    dAC = similar.(state.AL);
    
    S = spacetype(eltype(state));
    storage = storagetype(eltype(state));



    tvaltype_1_2 = TransposeFactType(S,storage,1,2);
    tvaltype_2_2 = TransposeFactType(S,storage,2,2);
    mvaltype_2_2 = DelayedFactType(S,storage,2,2);
    mvaltype_2_1 = DelayedFactType(S,storage,2,1);
    tvaltype_3_0 = TransposeFactType(S,storage,3,0);
    tvaltype_0_3 = TransposeFactType(S,storage,0,3);

    tfactory_1_2 = Dict{Any,tvaltype_1_2}();
    tfactory_2_2 = Dict{Any,tvaltype_2_2}();
    tfactory_3_0 = Dict{Any,tvaltype_3_0}();
    tfactory_0_3 = Dict{Any,tvaltype_0_3}();
    mfactory_2_2 = Dict{Any,mvaltype_2_2}();
    mfactory_2_1 = Dict{Any,mvaltype_2_1}();

    le = leftenv(envs,1,state);
    re = rightenv(envs,1,state);
    ac = state.AC[1];
    ac_flipped = transpose(ac',(1,3),(2,));

    function lo(leftind,e)
        if leftind == 0
            return 0
        end

        l = le[leftind];
        
        temp_0 = tfactory_1_2[(codomain(l)←domain(l),(1,),(3,2))];
        l_perm = temp_0(l);
        temp_1 = mfactory_2_2[codomain(ac_flipped)←domain(l_perm)];
        lAb = temp_1();

        mul!(lAb,ac_flipped,l_perm);
        free!(temp_0,l_perm);

        temp_2 = tfactory_2_2[(codomain(lAb)←domain(lAb),(3,1),(4,2))];
        lAb_perm = temp_2(lAb);

        free!(temp_1,lAb)

        temp_3 = mfactory_2_2[codomain(lAb_perm)←domain(e)];
        lAbe = temp_3();
        mul!(lAbe,lAb_perm,e);

        free!(temp_2,lAb_perm)

        temp_4 = tfactory_2_2[(codomain(lAbe)←domain(lAbe),(2,4),(1,3))]
        lAbe_perm = temp_4(lAbe);

        free!(temp_3,lAbe)

        temp_5 = mfactory_2_1[codomain(lAbe_perm)←domain(ac)]
        nl = temp_5();
        mul!(nl,lAbe_perm,ac);
        free!(temp_4,lAbe_perm)

        return nl
    end

    function or(e,rightind)
        if rightind == 0
            return 0
        end

        r = re[rightind];

        rt = tfactory_1_2[(codomain(r) ← domain(r),(1,),(3,2))](r)

        ar = mfactory_2_2[codomain(ac)←domain(rt)]()
        mul!(ar,ac,rt);
        
        free!(tfactory_1_2[(codomain(r) ← domain(r),(1,),(3,2))],rt)

        ar_t = tfactory_2_2[(codomain(ar)←domain(ar),(2,4),(1,3))](ar)
        ear = mfactory_2_2[codomain(e)←domain(ar_t)]();
        mul!(ear,e,ar_t);


        free!(mfactory_2_2[codomain(ac)←domain(rt)],ar);

        ear_t = tfactory_2_2[(codomain(ear)←domain(ear),(3,1),(4,2))](ear);
        free!(tfactory_2_2[(codomain(ar)←domain(ar),(2,4),(1,3))],ar_t)

        nr = mfactory_2_1[codomain(ear_t)←domain(ac_flipped)]()
        mul!(nr,ear_t,ac_flipped)
        free!(tfactory_2_2[(codomain(ear)←domain(ear),(3,1),(4,2))],ear_t)

        nr
    end

    function lr(l,r)
        if l == 0 || r == 0
            return zero(eltype(ac))
        end

        if l isa Int
            lvec = le[l]
        else
            lvec = l;
        end

        if r isa Int
            rvec = re[r]
        else
            rvec = r;
        end
        f_l = tfactory_0_3[codomain(lvec)←domain(lvec),(),(3,2,1)];
        f_r = tfactory_3_0[codomain(rvec)←domain(rvec),(1,2,3),()];
        r_trans = f_r(rvec)
        l_trans = f_l(lvec)
        toret = tr(l_trans*r_trans)
        free!(f_l,l_trans)
        free!(f_r,r_trans)
        return toret
    end

    function fast_expval(leftind,rightind,opp)
        if leftind == 0 || rightind == 0
            return zero(eltype(ac))
        end
        
        left = lo(leftind,opp)
        toret = lr(left,rightind)
        free!(mfactory_2_1[codomain(left)←domain(left)],left)
        return toret
    end

    for loc in 1:basis_size
        indmap_1L = indmap_1Ls[loc]
        indmap_2L = indmap_2Ls[loc]
        indmap_1R = indmap_1Rs[loc+1]
        indmap_2R = indmap_2Rs[loc+1]


        ac = state.AC[loc];
        ac_flipped = transpose(ac',(1,3),(2,));

        le = leftenv(envs,loc,state);
        re = rightenv(envs,loc,state);
        l_end = length(le);
        r_end = length(re);

        (tfactory_2_2, mfactory_2_2, mfactory_2_1,tfactory_1_2, tfactory_0_3, tfactory_3_0) = _make_dVdK_factories(ac);
        
        dAC[loc] = MPSKit.∂∂AC(loc, state, qchemham, envs)(ac);

        # onsite
        let
            expv = fast_expval(1,r_end,add_util_leg(h_pm));
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

            expv = fast_expval(1,r_end,add_util_leg(h_ppmm));
            dV[loc,loc,loc,loc] += expv;
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
                        dK[loc,i] += lr(indmap_1L[2,i],r_1)
                        dK[i,loc] += lr(indmap_1L[1,i],r_2)

                        expv = lr(indmap_1L[2,i],r_3);
                        dV[loc,loc,i,loc] += expv;
                        dV[loc,loc,loc,i] += expv;

                        expv = lr(indmap_1L[1,i],r_4);
                        dV[loc,i,loc,loc] += expv;
                        dV[i,loc,loc,loc] += expv;
                    end
                end
            end

            # ---

            let
                l_1 = lo(1,ppRm);
                l_2 = lo(1,Rpmm);
                for j in loc+1:basis_size      
                    expv = lr(l_1,indmap_1R[2,j]);
                    dV[loc,loc,j,loc] += expv;
                    dV[loc,loc,loc,j] += expv;

                    expv = lr(l_2,indmap_1R[1,j]);
                    dV[j,loc,loc,loc] += expv;
                    dV[loc,j,loc,loc] += expv;
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
                        dV[k,i,j,j] += expv;
                        dV[i,k,j,j] += expv

                        expv = lr(l_2,indmap_1R[2,k]);
                        dV[j,j,k,i] += expv;
                        dV[j,j,i,k] += expv;

                        expv = lr(l_3,indmap_1R[2,k]);
                        dV[j,i,j,k] += expv;
                        dV[i,j,k,j] += expv;

                        expv = lr(l_4,indmap_1R[2,k]);
                        dV[j,i,k,j] += expv;
                        dV[i,j,j,k] += expv;

                        expv = lr(l_5,indmap_1R[1,k]);
                        dV[j,k,j,i] += expv;
                        dV[k,j,i,j] += expv;

                        expv = lr(l_6,indmap_1R[1,k]);
                        dV[j,k,i,j] += expv;
                        dV[k,j,j,i] += expv;
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
                        dV[i,i,j,k] += expv;
                        dV[i,i,k,j] += expv
                        
                        expv = lr(l_2,indmap_1R[2,k]);
                        dV[j,i,i,k] -= 2*expv;
                        dV[i,j,k,i] -= 2*expv;

                        expv = lr(l_3,indmap_1R[2,k]);
                        dV[i,j,i,k] += expv;
                        dV[j,i,k,i] += expv;

                        expv = lr(l_4,indmap_1R[1,k]);
                        dV[i,k,i,j] += 2*expv;
                        dV[k,i,j,i] += 2*expv;
                        dV[i,k,j,i] -= 2*expv;
                        dV[k,i,i,j] -= 2*expv;
                        
                        expv = lr(l_5,indmap_1R[1,k]);
                        dV[i,k,i,j] -= expv;
                        dV[k,i,j,i] -= expv;

                        expv = lr(l_6,indmap_1R[1,k]);
                        dV[j,k,i,i] += expv;
                        dV[k,j,i,i] += expv;
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
                        dV[k,i,j,j]+=expv;
                        dV[i,k,j,j]+=expv;

                        expv = lr(indmap_1L[2,i],r_2);
                        dV[j,j,k,i]+=expv;
                        dV[j,j,i,k]+=expv;

                        expv = lr(indmap_1L[1,i],r_3);
                        dV[j,i,j,k]+=expv;
                        dV[i,j,k,j]+=expv;

                        expv = lr(indmap_1L[1,i],r_4);
                        dV[j,i,k,j]+=expv;
                        dV[i,j,j,k]+=expv;

                        expv = lr(indmap_1L[2,i],r_5);
                        dV[j,k,j,i]+=expv;
                        dV[k,j,i,j]+=expv;

                        expv = lr(indmap_1L[2,i],r_6);
                        dV[j,k,i,j]+=expv;
                        dV[k,j,j,i]+=expv;

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
                        dV[j,i,k,k] +=expv;
                        dV[i,j,k,k] +=expv;

                        expv = lr(indmap_1L[2,i],r_2);
                        dV[k,k,j,i]+=expv;
                        dV[k,k,i,j]+=expv;

                        expv = lr(indmap_1L[2,i],r_3);
                        dV[j,k,i,k]+=expv;
                        dV[k,j,k,i]+=expv;
                        dV[k,j,i,k]-=2*expv;
                        dV[j,k,k,i]-=2*expv;

                        expv = lr(indmap_1L[2,i],r_4);
                        dV[j,k,i,k]+=expv;
                        dV[k,j,k,i]+=expv;

                        expv = lr(indmap_1L[1,i],r_5);
                        dV[i,k,j,k]+=expv;
                        dV[k,i,k,j]+=expv;
                        dV[i,k,k,j]-=2*expv;
                        dV[k,i,j,k]-=2*expv;
                        
                        expv = lr(indmap_1L[1,i],r_6);
                        dV[i,k,j,k]+=expv;
                        dV[k,i,k,j]+=expv;
                    
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
                            dV[i,i,k,k] += lr(indmap_2L[1,i,1,i],r_1);
                            dV[k,k,i,i] += lr(indmap_2L[2,i,2,i],r_2);
                            
                            expv = lr(indmap_2L[2,i,1,i],r_3);
                            dV[k,i,k,i]+=expv;
                            dV[i,k,i,k]+=expv;

                            expv = lr(indmap_2L[2,i,1,i],r_4);
                            dV[k,i,i,k]+=expv;
                            dV[i,k,k,i]+=expv;
                        end     
                    end

                    for i in 1:k-1,j in i+1:k-1
                        @Threads.spawn begin
                            expv = lr(indmap_2L[1,i,1,j],r_1);
                            dV[i,j,k,k]+=expv;
                            dV[j,i,k,k]+=expv;

                            expv = lr(indmap_2L[2,i,2,j],r_2);
                            dV[k,k,j,i]+=expv;
                            dV[k,k,i,j]+=expv;

                            expv = lr(indmap_2L[2,i,1,j],r_4);
                            dV[k,j,k,i]-=expv;
                            dV[j,k,i,k]-=expv;
                            dV[j,k,k,i]+=expv;
                            dV[k,j,i,k]+=expv;

                            expv = lr(indmap_2L[1,i,2,j],r_4);
                            dV[i,k,k,j]+=expv;
                            dV[k,i,j,k]+=expv;

                            expv = lr(indmap_2L[1,i,2,j],r_5);
                            dV[k,i,k,j]+=expv;
                            dV[i,k,j,k]+=expv;

                            expv = lr(indmap_2L[2,i,1,j],r_5);
                            dV[k,j,k,i]-=expv;
                            dV[j,k,i,k]-=expv;
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
                            dV[j,j,i,i] += lr(l_1,indmap_2R[1,j,1,j])
                            dV[i,i,j,j] += lr(l_2,indmap_2R[2,j,2,j]);

                            expv = lr(l_3,indmap_2R[2,j,1,j]);
                            dV[i,j,i,j]+=expv;
                            dV[j,i,j,i]+=expv;

                            expv = lr(l_4,indmap_2R[2,j,1,j]);
                            dV[j,i,i,j] += expv;
                            dV[i,j,j,i]+=expv;
                        end
                    end
                    for j in i+1:basis_size,k in j+1:basis_size
                        @Threads.spawn begin
                            expv = lr(l_1,indmap_2R[1,j,1,k])
                            dV[j,k,i,i] += expv;
                            dV[k,j,i,i] += expv;

                            expv = lr(l_2,indmap_2R[2,j,2,k]);
                            dV[i,i,j,k] += expv;
                            dV[i,i,k,j] += expv;

                            expv = lr(l_4,indmap_2R[2,j,1,k]);
                            dV[i,k,i,j] -= expv;
                            dV[k,i,j,i] -= expv;
                            dV[k,i,i,j] += expv;
                            dV[i,k,j,i] += expv;

                            expv = lr(l_4,indmap_2R[1,j,2,k]);
                            dV[i,j,k,i] += expv;
                            dV[j,i,i,k] += expv;

                            expv = lr(l_5,indmap_2R[1,j,2,k]);
                            dV[j,i,k,i] += expv;
                            dV[i,j,i,k] += expv;

                            expv = lr(l_5,indmap_2R[2,j,1,k]);
                            dV[i,k,i,j] -= expv;
                            dV[k,i,j,i] -= expv;
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

                            dV[j,k,l,i] -= 2*expv;
                            dV[k,j,i,l] -= 2*expv;

                            expv = lr(indmap_2L[1,i,2,j],r_1);

                            dV[i,k,l,j] -= 2*expv;
                            dV[k,i,j,l] -= 2*expv;
                            dV[k,i,l,j] += 2*expv;
                            dV[i,k,j,l] += 2*expv

                            
                            expv = lr(indmap_2L[2,i,1,j],r_2);
                            dV[k,j,l,i] += expv;
                            dV[j,k,i,l] += expv;


                            expv = lr(indmap_2L[1,i,2,j],r_2);
                            dV[k,i,l,j] -= expv;
                            dV[i,k,j,l] -= expv;

                            expv = lr(indmap_2L[2,i,1,j],r_3)
                            dV[j,l,k,i]-=2*expv;
                            dV[l,j,i,k]-=2*expv;
                            dV[l,j,k,i]+=2*expv;
                            dV[j,l,i,k]+=2*expv;
                            
                            expv = lr(indmap_2L[1,i,2,j],r_3)
                            dV[i,l,k,j]-=2*expv;
                            dV[l,i,j,k]-=2*expv;


                            expv = lr(indmap_2L[2,i,1,j],r_4)
                            dV[l,j,k,i] -= expv;
                            dV[j,l,i,k] -= expv;
                        
                            expv = lr(indmap_2L[1,i,2,j],r_4)
                            dV[l,i,k,j] += expv;
                            dV[i,l,j,k] += expv;
                            
                            expv = lr(indmap_2L[1,i,1,j],r_5)
                            dV[j,i,l,k] += 2*expv;
                            dV[i,j,k,l] += 2*expv;


                            expv = lr(indmap_2L[1,i,1,j],r_6)
                            dV[i,j,l,k] += expv;
                            dV[j,i,k,l] += expv;
                            dV[j,i,l,k] -= expv;
                            dV[i,j,k,l] -= expv;

                            expv = lr(indmap_2L[2,i,2,j],r_7)
                            dV[l,k,j,i] += expv;
                            dV[k,l,i,j] += expv;

                            expv = lr(indmap_2L[2,i,2,j],r_8)
                            dV[l,k,i,j] += expv;
                            dV[k,l,j,i] += expv;
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
                            dV[j,k,l,i]-=2*expv;
                            dV[k,j,i,l]-=2*expv;
                        
                            expv = lr(l_2,indmap_1R[2,l]);
                            dV[i,k,l,j]-=2*expv;
                            dV[k,i,j,l]-=2*expv;
                            dV[k,i,l,j]+=2*expv;
                            dV[i,k,j,l]+=2*expv;

                            expv = lr(l_3,indmap_1R[2,l]);
                            dV[k,j,l,i]+=expv;
                            dV[j,k,i,l]+=expv;

                            expv = lr(l_4,indmap_1R[2,l]);
                            dV[k,i,l,j]-=expv;
                            dV[i,k,j,l]-=expv;


                            expv = lr(l_5,indmap_1R[1,l]);
                            dV[j,l,k,i]-=2*expv;
                            dV[l,j,i,k]-=2*expv;
                            dV[l,j,k,i]+=2*expv;
                            dV[j,l,i,k]+=2*expv;

                            expv = lr(l_6,indmap_1R[1,l]);
                            dV[l,j,k,i]-=expv;
                            dV[j,l,i,k]-=expv;

                            expv = lr(l_7,indmap_1R[1,l]);
                            dV[i,l,k,j]-=2*expv;
                            dV[l,i,j,k]-=2*expv;


                            expv = lr(l_8,indmap_1R[1,l]);
                            dV[l,i,k,j]+=expv;
                            dV[i,l,j,k]+=expv;


                            expv = lr(l_9,indmap_1R[2,l]);
                            dV[j,i,l,k]+=2*expv;
                            dV[i,j,k,l]+=2*expv;

                            expv = lr(l_10,indmap_1R[2,l]);
                            dV[i,j,l,k]+=expv;
                            dV[j,i,k,l]+=expv;
                            dV[j,i,l,k]-=expv;
                            dV[i,j,k,l]-=expv;

                            expv = lr(l_11,indmap_1R[1,l]);
                            dV[l,k,j,i]+=expv;
                            dV[k,l,i,j]+=expv;


                            expv = lr(l_12,indmap_1R[1,l]);
                            dV[l,k,i,j]+=expv;
                            dV[k,l,j,i]+=expv;
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
                            dV[j,k,i,l]-=expv/2
                            dV[k,j,l,i]-=expv/2
                            dV[k,j,i,l]+=expv;
                            dV[j,k,l,i]+=expv;

                            expv = lr(l_1,indmap_2R[2,k,1,l]);
                            dV[l,j,k,i]-=expv/2
                            dV[j,l,i,k]-=expv/2
                            dV[l,j,i,k]+=expv;
                            dV[j,l,k,i]+=expv;

                            expv = lr(l_2,indmap_2R[1,k,2,l]);
                            dV[j,k,i,l]+=expv;
                            dV[k,j,l,i]+=expv;
                            
                            expv = lr(l_2,indmap_2R[2,k,1,l]);
                            dV[l,j,k,i]-=expv;
                            dV[j,l,i,k]-=expv;

                            expv = lr(l_3,indmap_2R[1,k,2,l],);
                            dV[i,k,j,l]-=expv/2
                            dV[k,i,l,j]-=expv/2
                            dV[k,i,j,l]+=expv;
                            dV[i,k,l,j]+=expv;

                            expv = lr(l_3,indmap_2R[2,k,1,l]);
                            dV[l,i,k,j]-=expv/2
                            dV[i,l,j,k]-=expv/2
                            dV[l,i,j,k]+=expv;
                            dV[i,l,k,j]+=expv;

                            expv = lr(l_4,indmap_2R[1,k,2,l]);
                            dV[i,k,j,l]+=expv;
                            dV[k,i,l,j]+=expv;

                            expv = lr(l_4,indmap_2R[2,k,1,l]);
                            dV[l,i,k,j]-=expv;
                            dV[i,l,j,k]-=expv;

                            expv = lr(l_5,indmap_2R[1,k,1,l]);
                            dV[l,k,j,i]+=expv;
                            dV[k,l,i,j]+=expv;
                            dV[l,k,i,j]+=expv;
                            dV[k,l,j,i]+=expv;

                            expv = lr(l_6,indmap_2R[1,k,1,l]);
                            dV[l,k,j,i]-=expv;
                            dV[k,l,i,j]-=expv;
                            dV[l,k,i,j]+=expv;
                            dV[k,l,j,i]+=expv;

                            expv = lr(l_7,indmap_2R[2,k,2,l]);
                            dV[j,i,k,l]+=expv;
                            dV[i,j,l,k]+=expv;
                            dV[i,j,k,l]+=expv;
                            dV[j,i,l,k]+=expv;

                            expv = lr(l_8,indmap_2R[2,k,2,l]);
                            dV[j,i,k,l]+=expv;
                            dV[i,j,l,k]+=expv;
                            dV[i,j,k,l]-=expv;
                            dV[j,i,l,k]-=expv;
                        end

                    end
                end

            end
                        
            #=
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
            =#

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
                            dV[j,k,i,l]+=expv;
                            dV[k,j,l,i]+=expv;
                            dV[k,j,i,l]-=2*expv;
                            dV[j,k,l,i]-=2*expv;

                            expv = lr(indmap_2L[1,i,2,j],r_1);
                            dV[k,i,j,l]-=2*expv;
                            dV[i,k,l,j]-=2*expv;
                            dV[i,k,j,l]+=expv;
                            dV[k,i,l,j]+=expv;

                            expv = lr(indmap_2L[1,i,2,j],r_2);
                            dV[l,i,k,j]+=expv;
                            dV[i,l,j,k]+=expv;
                            dV[l,i,j,k]-=2*expv;
                            dV[i,l,k,j]-=2*expv;

                            expv = lr(indmap_2L[2,i,1,j],r_2);
                            dV[l,j,k,i]+=expv;
                            dV[j,l,i,k]+=expv;
                            dV[l,j,i,k]-=2*expv;
                            dV[j,l,k,i]-=2*expv;


                            expv = lr(indmap_2L[1,i,1,j],r_3);
                            dV[j,i,k,l]+=expv;
                            dV[i,j,l,k]+=expv;
                            dV[i,j,k,l]-=expv;
                            dV[j,i,l,k]-=expv;

                            expv = lr(indmap_2L[2,i,2,j],r_4);
                            dV[l,k,i,j]+=expv;
                            dV[k,l,j,i]+=expv;
                            dV[l,k,j,i]-=expv;
                            dV[k,l,i,j]-=expv;


                            expv = lr(indmap_2L[2,i,1,j],r_5);
                            dV[k,j,i,l]+=2*expv;
                            dV[j,k,l,i]+=2*expv;

                            expv = lr(indmap_2L[1,i,2,j],r_5);
                            dV[k,i,j,l]+=2*expv;
                            dV[i,k,l,j]+=2*expv;
                            dV[i,k,j,l]-=2*expv;
                            dV[k,i,l,j]-=2*expv;

                            expv = lr(indmap_2L[1,i,2,j],r_6);
                            dV[l,i,j,k]+=2*expv;
                            dV[i,l,k,j]+=2*expv

                            expv = lr(indmap_2L[2,i,1,j],r_6);
                            dV[l,j,k,i]-=2*expv;
                            dV[j,l,i,k]-=2*expv;
                            dV[l,j,i,k]+=2*expv;
                            dV[j,l,k,i]+=2*expv;

                            expv = lr(indmap_2L[1,i,1,j],r_7);
                            dV[i,j,k,l]+=2*expv;
                            dV[j,i,l,k]+=2*expv;

                            expv = lr(indmap_2L[2,i,2,j],r_8);
                            dV[l,k,j,i]+=2*expv;
                            dV[k,l,i,j]+=2*expv;
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
                            dV[i,k,i,k]+=expv;
                            dV[k,i,k,i]+=expv

                            expv = lr(indmap_2L[2,i,1,i],r_1);
                            dV[i,k,k,i]-=2*expv;
                            dV[k,i,i,k]-=2*expv;

                            expv = lr(indmap_2L[2,i,1,i],r_2);
                            dV[i,k,k,i]+=2*expv;
                            dV[k,i,i,k]+=2*expv;

                            expv = lr(indmap_2L[2,i,2,i],r_3);
                            dV[k,k,i,i]+=expv;

                            expv = lr(indmap_2L[1,i,1,i],r_4);
                            dV[i,i,k,k]+=expv;
                        end
                    end

                    for i in 1:half_basis_size-1, j in i+1:half_basis_size-1
                        @Threads.spawn begin

                            expv = lr(indmap_2L[2,i,1,j],r_1);
                            dV[j,k,i,k]+=expv;
                            dV[k,j,k,i]+=expv;
                            dV[k,j,i,k]-=2*expv;
                            dV[j,k,k,i]-=2*expv;

                            expv = lr(indmap_2L[1,i,2,j],r_1);
                            dV[i,k,j,k]+=expv;
                            dV[k,i,k,j]+=expv;
                            dV[i,k,k,j]-=2*expv;
                            dV[k,i,j,k]-=2*expv

                            expv = lr(indmap_2L[2,i,1,j],r_2);
                            dV[j,k,i,k]-=2*expv;
                            dV[k,j,k,i]-=2*expv;
                            dV[k,j,i,k]+=2*expv;
                            dV[j,k,k,i]+=2*expv;

                            expv = lr(indmap_2L[1,i,2,j],r_2);
                            dV[i,k,k,j]+=2*expv;
                            dV[k,i,j,k]+=2*expv

                            expv = lr(indmap_2L[2,i,2,j],r_3);
                            dV[k,k,j,i]+=expv;
                            dV[k,k,i,j]+=expv;

                            expv = lr(indmap_2L[1,i,1,j],r_4);
                            dV[j,i,k,k]+=expv;
                            dV[i,j,k,k]+=expv;
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
                            dV[j,k,i,i]+=expv;
                            dV[k,j,i,i]+=expv;

                            expv = lr(l_2,indmap_2R[2,j,2,k]);
                            dV[i,i,j,k]+=expv;
                            dV[i,i,k,j]+=expv;

                            expv = lr(l_3,indmap_2R[1,j,2,k]);
                            dV[j,i,i,k]-=2*expv;
                            dV[i,j,k,i]-=2*expv;
                            dV[i,j,i,k]+=expv;
                            dV[j,i,k,i]+=expv;

                            expv = lr(l_3,indmap_2R[2,j,1,k]);
                            dV[k,i,i,j]-=2*expv;
                            dV[i,k,j,i]-=2*expv;
                            dV[k,i,j,i]+=expv;
                            dV[i,k,i,j]+=expv;

                            expv = lr(l_4,indmap_2R[1,j,2,k]);
                            dV[j,i,i,k]+=2*expv;
                            dV[i,j,k,i]+=2*expv;

                            expv = lr(l_4,indmap_2R[2,j,1,k]);
                            dV[k,i,i,j]+=2*expv;
                            dV[i,k,j,i]+=2*expv;
                            dV[k,i,j,i]-=2*expv;
                            dV[i,k,i,j]-=2*expv

                        end
                    end
                end

            end
        
        end
        
        
       
    end

    energy = real(sum(expectation_value(state, qchemham, envs)))
    #@show energy
    (dV,dK,dAC,energy)
end

function project_U(odV,odK,h::CASSCF_Ham)
    #=
    energy should be V * odV + K * odK + E 
    odV
    =#

    g = zero(h.ao2mo);

    B = h.ao2mo#vecs*diagm(sqrt.(vals).^-1)*h.U'
    
    K = B'*(nuclear(h.basis)+kinetic(h.basis))*B;
    #c_k += K[h.active,h.active]
    K_a_o = K[h.active,:];
    @tensor g[h.active,:][-1;-2] += -K_a_o[1;-2]*odK[1;-1];
    K_o_a = K[:,h.active];
    @tensor g[:,h.active][-1;-2] += K_o_a[-1;1]*odK[-2;1];

    filled = 1:(h.active.start-1);
    active = h.active;

    Nvals = GaussianBasis.num_basis.(h.basis.basis)
    ao_offset = [sum(Nvals[1:(i-1)])  for i = 1:h.basis.nshells]

    #V_nuc += tr(K[filled,filled])*2
    k_o_f = K[:,filled];
    k_f_o = K[filled,:];
    
    g[filled,:] -= k_f_o*2;
    g[:,filled] += k_o_f*2;
   
    
    buffers = typeof(g)[];
    
    ijkls = Tuple{Int,Int,Int,Int}[];
    for i in 1:h.basis.nshells, j in 1:i, k in 1:i, l in 1:k
        k!=i || j>=l || continue; 
        push!(ijkls,(i,j,k,l))
    end
    
    @sync for sub_ijkls in partition(ijkls,Int(round(length(ijkls)/nthreads())))
        push!(buffers,zero(g));
        tcur_buff = buffers[end];

        @Threads.spawn begin
            for (i,j,k,l) in sub_ijkls
                
                buff = ERI_2e4c(h.basis,i,j,k,l)/2;
                norm(buff)<1e-12 && continue
                
                eightfold_way = unique([(1,2,3,4),(2,1,3,4),(2,1,4,3),(1,2,4,3),(3,4,1,2),(3,4,2,1),(4,3,2,1),(4,3,1,2)]) do (ai,bi,ci,di)
                    (a,b,c,d) = map(x->(i,j,k,l)[x],(ai,bi,ci,di))
                end
                
                buff /= 8/length(eightfold_way);
                
                # can be sped up by a factor of 2 if U is known to be real
                for (ai,bi,ci,di) in [(1,2,3,4),(2,1,3,4)]    
                    (a,b,c,d) = map(x->(i,j,k,l)[x],(ai,bi,ci,di))
                    buff_perm = permutedims(buff,(ai,bi,ci,di));

                    cur_buff = zero(tcur_buff);

                    sl_a = ao_offset[a]+1:ao_offset[a]+Nvals[a];
                    sl_b = ao_offset[b]+1:ao_offset[b]+Nvals[b];
                    sl_c = ao_offset[c]+1:ao_offset[c]+Nvals[c];
                    sl_d = ao_offset[d]+1:ao_offset[d]+Nvals[d];
                    t_1 = @view B'[active,sl_a]; t_1_p = @view B'[:,sl_a];
                    t_2 = @view B'[active,sl_c]; t_2_p = @view B'[:,sl_c];
                    t_3 = @view B[sl_d,active]; t_3_p = @view B[sl_d,:];
                    t_4 = @view B[sl_b,active]; t_4_p = @view B[sl_b,:];
                    #@tensor g[-1;-2] := t_1[5,1]*t_2[6,2]*t_3[3,7]*t_4[4,8]*buff_perm[1,4,2,3]*odV[5,6,7,8]
                    @tensor cur_buff[:,active][-1;-2] += t_1_p[-1,1]*t_2[6,2]*t_3[3,7]*t_4[4,8]*buff_perm[1,4,2,3]*odV[-2,6,7,8] order = (8, 3, 2, 6, 7, 4, 1)
                    @tensor cur_buff[:,active][-1;-2] += t_1[5,1]*t_2_p[-1,2]*t_3[3,7]*t_4[4,8]*buff_perm[1,4,2,3]*odV[5,-2,7,8] order = (8, 3, 1, 5, 7, 4, 2)
                    @tensor cur_buff[active,:][-1;-2] -= t_1[5,1]*t_2[6,2]*t_3_p[3,-2]*t_4[4,8]*buff_perm[1,4,2,3]*odV[5,6,-1,8] order = (8, 2, 1, 5, 6, 4, 3)
                    @tensor cur_buff[active,:][-1;-2] -= t_1[5,1]*t_2[6,2]*t_3[3,7]*t_4_p[4,-2]*buff_perm[1,4,2,3]*odV[5,6,7,-1] order = (1, 7, 6, 2, 3, 5, 4)
                    
                    t_1 = @view B'[active,sl_a];
                    t_2 = @view B'[filled,sl_c]
                    t_3 = @view B[sl_d,filled]
                    t_4 = @view B[sl_b,active]
                    @tensor cur_buff[:,active][-1;-2] += odK[-2,7]*t_1_p[-1,1]*t_2[2,3]*t_3[4,2]*t_4[5,7]*2*buff_perm[1,5,3,4] order = (2, 3, 4, 5, 1, 7)
                    @tensor cur_buff[:,filled][-1;-2] += odK[6,7]*t_1[6,1]*t_2_p[-1,3]*t_3[4,-2]*t_4[5,7]*2*buff_perm[1,5,3,4] order = (7, 6, 1, 5, 4, 3)
                    @tensor cur_buff[filled,:][-1;-2] -= odK[6,7]*t_1[6,1]*t_2[-1,3]*t_3_p[4,-2]*t_4[5,7]*2*buff_perm[1,5,3,4] order = (7, 6, 1, 5, 3, 4)
                    @tensor cur_buff[active,:][-1;-2] -= odK[6,-1]*t_1[6,1]*t_2[2,3]*t_3[4,2]*t_4_p[5,-2]*2*buff_perm[1,5,3,4] order = (2, 3, 4, 1, 5, 6)
                            
                    t_1 = @view B'[filled,sl_a];
                    t_2 = @view B'[active,sl_c]
                    t_3 = @view B[sl_d,active]
                    t_4 = @view B[sl_b,filled]
                    @tensor cur_buff[:,filled][-1;-2] += odK[6,7]*t_1_p[-1,1]*t_2[6,3]*t_3[4,7]*t_4[5,-2]*2*buff_perm[1,5,3,4] order = (7, 6, 3, 4, 5, 1)
                    @tensor cur_buff[:,active][-1;-2] += odK[-2,7]*t_1[2,1]*t_2_p[-1,3]*t_3[4,7]*t_4[5,2]*2*buff_perm[1,5,3,4] order = (7, 2, 1, 5, 4, 3)
                    @tensor cur_buff[active,:][-1;-2] -= odK[6,-1]*t_1[2,1]*t_2[6,3]*t_3_p[4,-2]*t_4[5,2]*2*buff_perm[1,5,3,4] order = (2, 1, 5, 3, 4, 6)
                    @tensor cur_buff[filled,:][-1;-2] -= odK[6,7]*t_1[-1,1]*t_2[6,3]*t_3[4,7]*t_4_p[5,-2]*2*buff_perm[1,5,3,4] order = (7, 6, 3, 4, 1, 5)
                            
                    t_1 = @view B'[filled,sl_a];
                    t_2 = @view B'[active,sl_c]
                    t_3 = @view B[sl_d,filled]
                    t_4 = @view B[sl_b,active]
                    @tensor cur_buff[:,filled][-1;-2] -= odK[6,7]*t_1_p[-1,1]*t_2[6,3]*t_3[4,-2]*t_4[5,7]*buff_perm[1,5,3,4] order = (6, 7, 5, 3, 4, 1)
                    @tensor cur_buff[:,active][-1;-2] -= odK[-2,7]*t_1[2,1]*t_2_p[-1,3]*t_3[4,2]*t_4[5,7]*buff_perm[1,5,3,4] order = (2, 1, 4, 5, 3, 7)
                    @tensor cur_buff[filled,:][-1;-2] += odK[6,7]*t_1[-1,1]*t_2[6,3]*t_3_p[4,-2]*t_4[5,7]*buff_perm[1,5,3,4] order = (6, 7, 5, 3, 1, 4)
                    @tensor cur_buff[active,:][-1;-2] += odK[6,-1]*t_1[2,1]*t_2[6,3]*t_3[4,2]*t_4_p[5,-2]*buff_perm[1,5,3,4] order = (2, 1, 4, 3, 5, 6)
                            
                    t_1 = @view B'[active,sl_a];
                    t_2 = @view B'[filled,sl_c]
                    t_3 = @view B[sl_d,active]
                    t_4 = @view B[sl_b,filled]
                    @tensor cur_buff[:,active][-1;-2] -= odK[-2,7]*t_1_p[-1,1]*t_2[2,3]*t_3[4,7]*t_4[5,2]*buff_perm[1,5,3,4] order = (2, 5, 3, 4, 1, 7)
                    @tensor cur_buff[:,filled][-1;-2] -= odK[6,7]*t_1[6,1]*t_2_p[-1,3]*t_3[4,7]*t_4[5,-2]*buff_perm[1,5,3,4] order = (6, 7, 1, 4, 5, 3)
                    @tensor cur_buff[active,:][-1;-2] += odK[6,-1]*t_1[6,1]*t_2[2,3]*t_3_p[4,-2]*t_4[5,2]*buff_perm[1,5,3,4] order = (2, 5, 3, 1, 4, 6)
                    @tensor cur_buff[filled,:][-1;-2] += odK[6,7]*t_1[6,1]*t_2[-1,3]*t_3[4,7]*t_4_p[5,-2]*buff_perm[1,5,3,4] order = (6, 7, 1, 4, 3, 5)
                            
                    t_1 = @view B'[filled,sl_a];
                    t_2 = @view B'[filled,sl_c];
                    t_3 = @view B[sl_d,filled];
                    t_4 = @view B[sl_b,filled];
                    @tensor cur_buff[:,filled][-1;-2] += 4*t_1_p[-1,1]*t_2[6,2]*t_3[3,6]*t_4[4,-2]*buff_perm[1,4,2,3] order = (6, 2, 3, 4, 1)
                    @tensor cur_buff[:,filled][-1;-2] += 4*t_1[5,1]*t_2_p[-1,2]*t_3[3,-2]*t_4[4,5]*buff_perm[1,4,2,3] order = (5, 1, 4, 3, 2)
                    @tensor cur_buff[filled,:][-1;-2] -= 4*t_1[5,1]*t_2[-1,2]*t_3_p[3,-2]*t_4[4,5]*buff_perm[1,4,2,3] order = (5, 1, 4, 2, 3)
                    @tensor cur_buff[filled,:][-1;-2] -= 4*t_1[-1,1]*t_2[6,2]*t_3[3,6]*t_4_p[4,-2]*buff_perm[1,4,2,3] order = (6, 2, 3, 1, 4)
                            
                    @tensor cur_buff[:,filled][-1;-2] -= 2*t_1_p[-1,1]*t_2[6,2]*t_3[3,-2]*t_4[4,6]*buff_perm[1,4,2,3] order = (6, 4, 2, 3, 1)
                    @tensor cur_buff[:,filled][-1;-2] -= 2*t_1[5,1]*t_2_p[-1,2]*t_3[3,5]*t_4[4,-2]*buff_perm[1,4,2,3] order = (5, 1, 3, 4, 2)
                    @tensor cur_buff[filled,:][-1;-2] += 2*t_1[-1,1]*t_2[6,2]*t_3_p[3,-2]*t_4[4,6]*buff_perm[1,4,2,3] order = (6, 4, 2, 1, 3)
                    @tensor cur_buff[filled,:][-1;-2] += 2*t_1[5,1]*t_2[-1,2]*t_3[3,5]*t_4_p[4,-2]*buff_perm[1,4,2,3] order = (5, 1, 3, 2, 4)
                    
                    tcur_buff .+= cur_buff*2
                    tcur_buff .-= cur_buff'*2
                end
            end
        end
    end
    
    g += sum(buffers)
    
    gradient = -g;
    
    function apply_hessian(x)
        hess_g = zero(x);

        #K = B'*(nuclear(h.basis)+kinetic(h.basis))*B;
        hess_K = x*K + K*x'
        K_a_o = hess_K[h.active,:];
        @tensor hess_g[h.active,:][-1;-2] += -K_a_o[1;-2]*odK[1;-1];
        K_o_a = hess_K[:,h.active];
        @tensor hess_g[:,h.active][-1;-2] += K_o_a[-1;1]*odK[-2;1];

        k_o_f = hess_K[:,filled];
        k_f_o = hess_K[filled,:];
        
        hess_g[filled,:] -= k_f_o*2;
        hess_g[:,filled] += k_o_f*2;

        
        hess_buffers = typeof(hess_g)[];
        #=
        @sync for sub_ijkls in partition(ijkls,Int(round(length(ijkls)/nthreads())))
            push!(hess_buffers,zero(hess_g));
            tcur_buff = hess_buffers[end];

            @Threads.spawn begin
                for (i,j,k,l) in sub_ijkls

                    buff = ERI_2e4c(h.basis,i,j,k,l)/2;
                    norm(buff)<1e-12 && continue

                    eightfold_way = unique([(1,2,3,4),(2,1,3,4),(2,1,4,3),(1,2,4,3),(3,4,1,2),(3,4,2,1),(4,3,2,1),(4,3,1,2)]) do (ai,bi,ci,di)
                        (a,b,c,d) = map(x->(i,j,k,l)[x],(ai,bi,ci,di))
                    end
                    
                    buff /= 8/length(eightfold_way);
                    
                    for (ai,bi,ci,di) in [(1,2,3,4),(2,1,3,4)]
                        

                        (a,b,c,d) = map(x->(i,j,k,l)[x],(ai,bi,ci,di))
                        buff_perm = permutedims(buff,(ai,bi,ci,di));

                        cur_buff = zero(tcur_buff);
                        for (B1,B2,B3,B4) in [(x*B',B',B,B),(B',x*B',B,B),(B',B',B*x',B*x')]
                            
                            sl_a = ao_offset[a]+1:ao_offset[a]+Nvals[a];
                            sl_b = ao_offset[b]+1:ao_offset[b]+Nvals[b];
                            sl_c = ao_offset[c]+1:ao_offset[c]+Nvals[c];
                            sl_d = ao_offset[d]+1:ao_offset[d]+Nvals[d];
                            t_1 = @view B1[active,sl_a]; t_1_p = @view B1[:,sl_a];
                            t_2 = @view B2[active,sl_c]; t_2_p = @view B2[:,sl_c];
                            t_3 = @view B3[sl_d,active]; t_3_p = @view B3[sl_d,:];
                            t_4 = @view B4[sl_b,active]; t_4_p = @view B4[sl_b,:];
                            #@tensor g[-1;-2] := t_1[5,1]*t_2[6,2]*t_3[3,7]*t_4[4,8]*buff_perm[1,4,2,3]*odV[5,6,7,8]
                            @tensor cur_buff[:,active][-1;-2] += t_1_p[-1,1]*t_2[6,2]*t_3[3,7]*t_4[4,8]*buff_perm[1,4,2,3]*odV[-2,6,7,8] order = (8, 3, 2, 6, 7, 4, 1)
                            @tensor cur_buff[:,active][-1;-2] += t_1[5,1]*t_2_p[-1,2]*t_3[3,7]*t_4[4,8]*buff_perm[1,4,2,3]*odV[5,-2,7,8] order = (8, 3, 1, 5, 7, 4, 2)
                            @tensor cur_buff[active,:][-1;-2] -= t_1[5,1]*t_2[6,2]*t_3_p[3,-2]*t_4[4,8]*buff_perm[1,4,2,3]*odV[5,6,-1,8] order = (8, 2, 1, 5, 6, 4, 3)
                            @tensor cur_buff[active,:][-1;-2] -= t_1[5,1]*t_2[6,2]*t_3[3,7]*t_4_p[4,-2]*buff_perm[1,4,2,3]*odV[5,6,7,-1] order = (1, 7, 6, 2, 3, 5, 4)
                            
                            
                            t_1 = @view B1[active,sl_a];
                            t_2 = @view B2[filled,sl_c]
                            t_3 = @view B3[sl_d,filled]
                            t_4 = @view B4[sl_b,active]
                            @tensor cur_buff[:,active][-1;-2] += odK[-2,7]*t_1_p[-1,1]*t_2[2,3]*t_3[4,2]*t_4[5,7]*2*buff_perm[1,5,3,4] order = (2, 3, 4, 5, 1, 7)
                            @tensor cur_buff[:,filled][-1;-2] += odK[6,7]*t_1[6,1]*t_2_p[-1,3]*t_3[4,-2]*t_4[5,7]*2*buff_perm[1,5,3,4] order = (7, 6, 1, 5, 4, 3)
                            @tensor cur_buff[filled,:][-1;-2] -= odK[6,7]*t_1[6,1]*t_2[-1,3]*t_3_p[4,-2]*t_4[5,7]*2*buff_perm[1,5,3,4] order = (7, 6, 1, 5, 3, 4)
                            @tensor cur_buff[active,:][-1;-2] -= odK[6,-1]*t_1[6,1]*t_2[2,3]*t_3[4,2]*t_4_p[5,-2]*2*buff_perm[1,5,3,4] order = (2, 3, 4, 1, 5, 6)
                                    
                            t_1 = @view B1[filled,sl_a];
                            t_2 = @view B2[active,sl_c]
                            t_3 = @view B3[sl_d,active]
                            t_4 = @view B4[sl_b,filled]
                            @tensor cur_buff[:,filled][-1;-2] += odK[6,7]*t_1_p[-1,1]*t_2[6,3]*t_3[4,7]*t_4[5,-2]*2*buff_perm[1,5,3,4] order = (7, 6, 3, 4, 5, 1)
                            @tensor cur_buff[:,active][-1;-2] += odK[-2,7]*t_1[2,1]*t_2_p[-1,3]*t_3[4,7]*t_4[5,2]*2*buff_perm[1,5,3,4] order = (7, 2, 1, 5, 4, 3)
                            @tensor cur_buff[active,:][-1;-2] -= odK[6,-1]*t_1[2,1]*t_2[6,3]*t_3_p[4,-2]*t_4[5,2]*2*buff_perm[1,5,3,4] order = (2, 1, 5, 3, 4, 6)
                            @tensor cur_buff[filled,:][-1;-2] -= odK[6,7]*t_1[-1,1]*t_2[6,3]*t_3[4,7]*t_4_p[5,-2]*2*buff_perm[1,5,3,4] order = (7, 6, 3, 4, 1, 5)
                                    
                            t_1 = @view B1[filled,sl_a];
                            t_2 = @view B2[active,sl_c]
                            t_3 = @view B3[sl_d,filled]
                            t_4 = @view B4[sl_b,active]
                            @tensor cur_buff[:,filled][-1;-2] -= odK[6,7]*t_1_p[-1,1]*t_2[6,3]*t_3[4,-2]*t_4[5,7]*buff_perm[1,5,3,4] order = (6, 7, 5, 3, 4, 1)
                            @tensor cur_buff[:,active][-1;-2] -= odK[-2,7]*t_1[2,1]*t_2_p[-1,3]*t_3[4,2]*t_4[5,7]*buff_perm[1,5,3,4] order = (2, 1, 4, 5, 3, 7)
                            @tensor cur_buff[filled,:][-1;-2] += odK[6,7]*t_1[-1,1]*t_2[6,3]*t_3_p[4,-2]*t_4[5,7]*buff_perm[1,5,3,4] order = (6, 7, 5, 3, 1, 4)
                            @tensor cur_buff[active,:][-1;-2] += odK[6,-1]*t_1[2,1]*t_2[6,3]*t_3[4,2]*t_4_p[5,-2]*buff_perm[1,5,3,4] order = (2, 1, 4, 3, 5, 6)
                                    
                            t_1 = @view B1[active,sl_a];
                            t_2 = @view B2[filled,sl_c]
                            t_3 = @view B3[sl_d,active]
                            t_4 = @view B4[sl_b,filled]
                            @tensor cur_buff[:,active][-1;-2] -= odK[-2,7]*t_1_p[-1,1]*t_2[2,3]*t_3[4,7]*t_4[5,2]*buff_perm[1,5,3,4] order = (2, 5, 3, 4, 1, 7)
                            @tensor cur_buff[:,filled][-1;-2] -= odK[6,7]*t_1[6,1]*t_2_p[-1,3]*t_3[4,7]*t_4[5,-2]*buff_perm[1,5,3,4] order = (6, 7, 1, 4, 5, 3)
                            @tensor cur_buff[active,:][-1;-2] += odK[6,-1]*t_1[6,1]*t_2[2,3]*t_3_p[4,-2]*t_4[5,2]*buff_perm[1,5,3,4] order = (2, 5, 3, 1, 4, 6)
                            @tensor cur_buff[filled,:][-1;-2] += odK[6,7]*t_1[6,1]*t_2[-1,3]*t_3[4,7]*t_4_p[5,-2]*buff_perm[1,5,3,4] order = (6, 7, 1, 4, 3, 5)
                                    
                            t_1 = @view B1[filled,sl_a];
                            t_2 = @view B2[filled,sl_c];
                            t_3 = @view B3[sl_d,filled];
                            t_4 = @view B4[sl_b,filled];
                            @tensor cur_buff[:,filled][-1;-2] += 4*t_1_p[-1,1]*t_2[6,2]*t_3[3,6]*t_4[4,-2]*buff_perm[1,4,2,3] order = (6, 2, 3, 4, 1)
                            @tensor cur_buff[:,filled][-1;-2] += 4*t_1[5,1]*t_2_p[-1,2]*t_3[3,-2]*t_4[4,5]*buff_perm[1,4,2,3] order = (5, 1, 4, 3, 2)
                            @tensor cur_buff[filled,:][-1;-2] -= 4*t_1[5,1]*t_2[-1,2]*t_3_p[3,-2]*t_4[4,5]*buff_perm[1,4,2,3] order = (5, 1, 4, 2, 3)
                            @tensor cur_buff[filled,:][-1;-2] -= 4*t_1[-1,1]*t_2[6,2]*t_3[3,6]*t_4_p[4,-2]*buff_perm[1,4,2,3] order = (6, 2, 3, 1, 4)
                                    
                            @tensor cur_buff[:,filled][-1;-2] -= 2*t_1_p[-1,1]*t_2[6,2]*t_3[3,-2]*t_4[4,6]*buff_perm[1,4,2,3] order = (6, 4, 2, 3, 1)
                            @tensor cur_buff[:,filled][-1;-2] -= 2*t_1[5,1]*t_2_p[-1,2]*t_3[3,5]*t_4[4,-2]*buff_perm[1,4,2,3] order = (5, 1, 3, 4, 2)
                            @tensor cur_buff[filled,:][-1;-2] += 2*t_1[-1,1]*t_2[6,2]*t_3_p[3,-2]*t_4[4,6]*buff_perm[1,4,2,3] order = (6, 4, 2, 1, 3)
                            @tensor cur_buff[filled,:][-1;-2] += 2*t_1[5,1]*t_2[-1,2]*t_3[3,5]*t_4_p[4,-2]*buff_perm[1,4,2,3] order = (5, 1, 3, 2, 4)

                            
                        end
                        tcur_buff .+= cur_buff*2;
                        tcur_buff .-= cur_buff'*2;
                    end
                end
            end
        end
        hess_g+= sum(hess_buffers);
        =#        

        return -hess_g

    end


    gradient,apply_hessian
end


function transform(h::CASSCF_Ham)
    B = h.ao2mo
    c_eri = zeros(eltype(B),length(h.active),length(h.active),length(h.active),length(h.active));
    c_k = zeros(eltype(B),length(h.active),length(h.active));

    V_nuc = GaussianBasis.Molecules.nuclear_repulsion(h.basis.atoms);

    K = B'*(nuclear(h.basis)+kinetic(h.basis))*B;
    c_k += K[h.active,h.active]

    filled = 1:(h.active.start-1);

    Nvals = GaussianBasis.num_basis.(h.basis.basis)
    ao_offset = [sum(Nvals[1:(i-1)])  for i = 1:h.basis.nshells]

    V_nuc += tr(K[filled,filled])*2
    
    
    buff_k = typeof(c_k)[];
    buff_eri = typeof(c_eri)[];
    buff_nuc = typeof(V_nuc)[];
    
    ijkls = Tuple{Int,Int,Int,Int}[];
    for i in 1:h.basis.nshells, j in 1:i, k in 1:i, l in 1:k
        k!=i || j>=l || continue; 
        push!(ijkls,(i,j,k,l))
    end
    
    @sync for sub_ijkls in partition(ijkls,Int(round(length(ijkls)/nthreads())))

        push!(buff_k,zero(c_k));
        push!(buff_eri,zero(c_eri));
        push!(buff_nuc,zero(V_nuc));
        cur_k = buff_k[end];
        cur_eri = buff_eri[end];
        buff_index = length(buff_nuc);

        @Threads.spawn begin

            for (i,j,k,l) in sub_ijkls

                buff = ERI_2e4c(h.basis,i,j,k,l)/2;
                norm(buff)<1e-12 && continue

                eightfold_way = unique([(1,2,3,4),(2,1,3,4),(2,1,4,3),(1,2,4,3),(3,4,1,2),(3,4,2,1),(4,3,2,1),(4,3,1,2)]) do (ai,bi,ci,di)
                    (a,b,c,d) = map(x->(i,j,k,l)[x],(ai,bi,ci,di))
                end
                
                buff /= 8/length(eightfold_way);

                # can be sped up by a factor of 2 if U is known to be real
                for (ai,bi,ci,di) in [(1,2,3,4),(2,1,3,4)]
                    
                    (a,b,c,d) = map(x->(i,j,k,l)[x],(ai,bi,ci,di))
                    buff_perm = permutedims(buff,(ai,bi,ci,di));

                    sl_a = ao_offset[a]+1:ao_offset[a]+Nvals[a];
                    sl_b = ao_offset[b]+1:ao_offset[b]+Nvals[b];
                    sl_c = ao_offset[c]+1:ao_offset[c]+Nvals[c];
                    sl_d = ao_offset[d]+1:ao_offset[d]+Nvals[d];
                    
                    cur_eri_term = zero(cur_eri);
                    cur_k_term = zero(cur_k);
                    cur_nuc_term = zero(buff_nuc[buff_index]);

                    t_1 = @view B'[h.active,sl_a];
                    t_2 = @view B'[h.active,sl_c];
                    t_3 = @view B[sl_d,h.active];
                    t_4 = @view B[sl_b,h.active];
                    @tensor cur_eri_term[-1 -2;-3 -4] += t_1[-1,1]*t_2[-2,2]*t_3[3,-3]*t_4[4,-4]*buff_perm[1,4,2,3]
                    
                    
                    # t_K += ERI[h.active,a,a,h.active]*2
                    t_1 = @view B'[h.active,sl_a];
                    t_2 = @view B'[filled,sl_c]
                    t_3 = @view B[sl_d,filled]
                    t_4 = @view B[sl_b,h.active]
                    @tensor cur_k_term[-1;-2] += t_1[-1,1]*t_2[2,3]*t_3[4,2]*t_4[5,-2]*2*buff_perm[1,5,3,4]
                    
                    
                    # t_K += ERI[a,h.active,h.active,a]*2
                    t_1 = @view B'[filled,sl_a];
                    t_2 = @view B'[h.active,sl_c]
                    t_3 = @view B[sl_d,h.active]
                    t_4 = @view B[sl_b,filled]
                    @tensor cur_k_term[-1;-2] += t_1[2,1]*t_2[-1,3]*t_3[4,-2]*t_4[5,2]*2*buff_perm[1,5,3,4]

                    # t_K -= ERI[a,h.active,a,h.active]
                    t_1 = @view B'[filled,sl_a];
                    t_2 = @view B'[h.active,sl_c]
                    t_3 = @view B[sl_d,filled]
                    t_4 = @view B[sl_b,h.active]
                    @tensor cur_k_term[-1;-2] -= t_1[2,1]*t_2[-1,3]*t_3[4,2]*t_4[5,-2]*buff_perm[1,5,3,4]
                    
                    
                    # t_K -= ERI[h.active,a,h.active,a]
                    t_1 = @view B'[h.active,sl_a];
                    t_2 = @view B'[filled,sl_c]
                    t_3 = @view B[sl_d,h.active]
                    t_4 = @view B[sl_b,filled]
                    @tensor cur_k_term[-1;-2] -= t_1[-1,1]*t_2[2,3]*t_3[4,-2]*t_4[5,2]*buff_perm[1,5,3,4]

                    
                    # E += 4*ERI[a,b,b,a]
                    t_1 = @view B'[filled,sl_a];
                    t_2 = @view B'[filled,sl_c];
                    t_3 = @view B[sl_d,filled];
                    t_4 = @view B[sl_b,filled];
                    cur_nuc_term += 4*@tensor t_1[5,1]*t_2[6,2]*t_3[3,6]*t_4[4,5]*buff_perm[1,4,2,3]
                    
                    # E -= 2*ERI[a,b,a,b]
                    cur_nuc_term -= 2*@tensor t_1[5,1]*t_2[6,2]*t_3[3,5]*t_4[4,6]*buff_perm[1,4,2,3]
                    
                    cur_eri .+= cur_eri_term;
                    cur_eri .+= permutedims(cur_eri_term,(2,1,4,3));
                    cur_eri .+= permutedims(conj.(cur_eri_term),(4,3,2,1))
                    cur_eri .+= permutedims(conj.(cur_eri_term),(3,4,1,2))
                    
                    cur_k .+= 2*cur_k_term
                    cur_k .+= 2*cur_k_term'
                    buff_nuc[buff_index] += cur_nuc_term*2
                    buff_nuc[buff_index] += cur_nuc_term'*2
                end

                
                
            end
            
        end
    end

    V_nuc += sum(buff_nuc);
    c_k += sum(buff_k);
    c_eri += sum(buff_eri);
    
    return (V_nuc,c_k,c_eri)

end

function mpo_representation(h::CASSCF_Ham)
    (E,K,ERI) = transform(h);    
    fused_quantum_chemistry_hamiltonian(E,K,ERI,ComplexF64)
end


function manifoldpoint(state,ham)
    (dV,dK,dAC,E) = quantum_chemistry_dV_dK(ham,state);
    (dU,hessian) = project_U(dV,dK,ham);

    g = (Grassmann.project.(dAC,state.AL).*0)

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
    new_U = ham.ao2mo*exp(alpha*grad_U)'

    newpoint = manifoldpoint(y,CASSCF_Ham(ham.basis,new_U,ham.active));
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
    # I could not get this preconditioner to behave; nevertheless this appears to be the suggested approach...
    # the problem appears to be that the linesearch refuses to actually take a step in that direction, despite it being a descent direction
    return v
    (g_prec,grad_U) = v;
    (state,ham,Rhoreg,g,dU,hessian,E) = x;
    
    function precfun(w)
        (x_re,x_im,x_2) = w.vecs;

        x_1 = x_re+1im*x_im;
        x_1 = (x_1 - x_1')/2
        y_1 = hessian(x_1) + dU*x_2[1]
        y_2 = real(dot(dU,x_1))

        y_1 = (y_1-y_1')/2
        RecursiveVec(real.(y_1),imag.(y_1),[y_2])
    end

    #small workaround for complex U...
    (vals,vecs) = eigsolve(precfun,RecursiveVec(real.(grad_U),imag.(grad_U),[1]),1,:SR,Arnoldi())
    (x_r,x_i,α)=first(vecs).vecs
    nsol = (x_r+x_i*1im)/α[1]
    nsol = (nsol-nsol')/2
   

    return (g_prec,-nsol)
end

