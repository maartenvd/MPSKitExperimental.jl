# in this file I did not exploit symmetries in the ERI

#=

I took mpskitmodel's implementation of the qchem hamiltonian (which is somewhat readable) and propped it into a fused_mpoham.
Code is now impossible to read, but essentially identical in spirit to mpskitmodel's implementation
maybe this can be automated, using something like coallesce?

=#


# x * o_1 = o_2
find_left_map(o_1,o_2) = (o_2*o_1')*pinv(o_1*o_1');

# o_1 * x = o_2
find_right_map(o_1,o_2) = pinv(o_1'*o_1)*o_1'*o_2

function fused_quantum_chemistry_hamiltonian(E0,K,V,Elt=eltype(V))
    basis_size = size(K,1);
    half_basis_size = Int(ceil((basis_size+1)/2));
    #@show half_basis_size
    # the phsyical space
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


    # ----------------------------------------------------------------------
    # Maps something easier to understand to the corresponding virtual index
    # ----------------------------------------------------------------------

    cnt = 1;
    indmap_1L = fill(0,2,basis_size);
    for i in 1:2, j in 1:basis_size
        cnt += 1
        indmap_1L[i,j] = cnt;
    end

    indmap_1R = fill(0,2,basis_size);
    for i in 1:2, j in 1:basis_size
        cnt += 1
        indmap_1R[i,j] = cnt;
    end
    

    indmap_2L = fill(0,2,basis_size,2,basis_size);
    indmap_2R = fill(0,2,basis_size,2,basis_size);
    for pm1 in 1:2, i in 1:half_basis_size, pm2 in 1:2, j in i:half_basis_size
        cnt += 1
        indmap_2L[pm1,i,pm2,j] = cnt;
        indmap_2R[pm1,end-j+1,pm2,end-i+1] = cnt
    end

    function masks(a,b)
        lmask = fill(false,cnt+1);lmask[a] = true; rmask = fill(false,cnt+1);rmask[b] = true;
        (lmask,rmask)
    end

    function block2masks(lblock,opp,rblock)
        lmask = map(l->abs(l)>1e-12,lblock);
        rmask = map(l->abs(l)>1e-12,rblock);
        (lmask,lblock[lmask],opp,rblock[rmask],rmask)
    end

    domspaces = fill(oneunit(psp),basis_size+1,cnt+1);
    O = tensormaptype(spacetype(ap),2,2,storagetype(ap));
    #B = Elt
    op_blocks = Vector{Vector{Tuple{Vector{Bool},Vector{Elt},O,Vector{Elt},Vector{Bool}}}}(undef,basis_size);
    scal_blocks = Vector{Vector{Tuple{Vector{Bool},Vector{Elt},Elt,Vector{Elt},Vector{Bool}}}}(undef,basis_size);

    for b in 1:basis_size
        op_blocks[b] = [];
        scal_blocks[b] = [];
    end

    (l1mask,r1mask) = masks(1,1);
    (l2mask,r2mask) = masks(cnt+1,cnt+1);
    bvec = Elt[Elt(1)];
    for b in 1:basis_size
        push!(scal_blocks[b],(l1mask,bvec,Elt(1),bvec,r1mask));
        push!(scal_blocks[b],(l2mask,bvec,Elt(1),bvec,r2mask));
    end

    # fill indmap_1L and indmap_1R
    ut = Tensor(ones,oneunit(psp));
    @plansor ut_ap[-1 -2;-3 -4] := ut[-1]*ap[-3 -2;-4];
    @plansor ut_am[-1 -2;-3 -4] := ut[-1]*am[-3 -2;-4];
    @plansor bp_ut[-1 -2;-3 -4] := bp[-1;-3 -2]*conj(ut[-4]);
    @plansor bm_ut[-1 -2;-3 -4] := bm[-1;-3 -2]*conj(ut[-4]);
    for i in 1:basis_size
        (lmask,rmask) = masks(1,indmap_1L[1,i]);
        push!(op_blocks[i],(lmask,Elt[Elt(1)],ut_ap,Elt[Elt(1)],rmask))

        (lmask,rmask) = masks(1,indmap_1L[2,i]);
        push!(op_blocks[i],(lmask,Elt[Elt(1)],ut_am,Elt[Elt(1)],rmask))

        (lmask,rmask) = masks(indmap_1R[1,i],cnt+1);
        push!(op_blocks[i],(lmask,Elt[Elt(1)],bp_ut,Elt[Elt(1)],rmask))

        (lmask,rmask) = masks(indmap_1R[2,i],cnt+1);
        push!(op_blocks[i],(lmask,Elt[Elt(1)],bm_ut,Elt[Elt(1)],rmask))
        
        for loc in i+1:basis_size
            (lmask,rmask) = masks(indmap_1L[1,i],indmap_1L[1,i]);
            push!(scal_blocks[loc],(lmask,Elt[Elt(1)],Elt(1),Elt[Elt(1)],rmask));

            (lmask,rmask) = masks(indmap_1L[2,i],indmap_1L[2,i]);
            push!(scal_blocks[loc],(lmask,Elt[Elt(1)],Elt(1),Elt[Elt(1)],rmask));
        end

        for loc in 1:i-1
            (lmask,rmask) = masks(indmap_1R[1,i],indmap_1R[1,i]);
            push!(scal_blocks[loc],(lmask,Elt[Elt(1)],Elt(1),Elt[Elt(1)],rmask));

            (lmask,rmask) = masks(indmap_1R[2,i],indmap_1R[2,i]);
            push!(scal_blocks[loc],(lmask,Elt[Elt(1)],Elt(1),Elt[Elt(1)],rmask));
        end
    end
    for loc in 1:basis_size+1, i in 1:basis_size
        domspaces[loc,indmap_1L[1,i]] = _lastspace(ut_ap)'
        domspaces[loc,indmap_1L[2,i]] = _lastspace(ut_am)'
        domspaces[loc,indmap_1R[1,i]] = _firstspace(bp_ut)
        domspaces[loc,indmap_1R[2,i]] = _firstspace(bm_ut)
    end
    # indmap_2 onsite part
    # we need pp, mm, pm
    
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

    @plansor ut_apap[-1 -2;-3 -4] := ut[-1]*ap[-3 1;3]*ap[1 -2;4]*conj(pp_f[-4;3 4]);
    @plansor ut_amam[-1 -2;-3 -4] := ut[-1]*am[-3 1;3]*am[1 -2;4]*conj(mm_f[-4;3 4]);
    @plansor ut_amap[-1 -2;-3 -4] := ut[-1]*am[-3 1;3]*ap[1 -2;4]*conj(mp_f[-4;3 4]);
    @plansor ut_apam[-1 -2;-3 -4] := ut[-1]*ap[-3 1;3]*am[1 -2;4]*conj(pm_f[-4;3 4]);
    
    @plansor bpbp_ut[-1 -2;-3 -4] := mm_f[-1;1 2]*bp[1;-3 3]*bp[2;3 -2]*conj(ut[-4]);
    @plansor bmbm_ut[-1 -2;-3 -4] := pp_f[-1;1 2]*bm[1;-3 3]*bm[2;3 -2]*conj(ut[-4]);
    @plansor bmbp_ut[-1 -2;-3 -4] := pm_f[-1;1 2]*bm[1;-3 3]*bp[2;3 -2]*conj(ut[-4]);
    @plansor bpbm_ut[-1 -2;-3 -4] := mp_f[-1;1 2]*bp[1;-3 3]*bm[2;3 -2]*conj(ut[-4]);
    
    for i in 1:basis_size
        if i < half_basis_size
            (lmask,rmask) = masks(1,indmap_2L[1,i,1,i]);
            push!(op_blocks[i],(lmask,Elt[Elt(1)],ut_apap,Elt[Elt(1)],rmask))
            
            (lmask,rmask) = masks(1,indmap_2L[2,i,1,i]);
            push!(op_blocks[i],(lmask,Elt[Elt(1)],ut_amap,Elt[Elt(1)],rmask))

            (lmask,rmask) = masks(1,indmap_2L[1,i,2,i]);
            push!(op_blocks[i],(lmask,Elt[Elt(1)],ut_apam,Elt[Elt(1)],rmask))

            (lmask,rmask) = masks(1,indmap_2L[2,i,2,i]);
            push!(op_blocks[i],(lmask,Elt[Elt(1)],ut_amam,Elt[Elt(1)],rmask))
            
            for loc in i+1:half_basis_size-1

                (lmask,rmask) = masks(indmap_2L[1,i,1,i],indmap_2L[1,i,1,i]);
                push!(scal_blocks[loc],(lmask,Elt[Elt(1)],Elt(1),Elt[Elt(1)],rmask));

                (lmask,rmask) = masks(indmap_2L[2,i,1,i],indmap_2L[2,i,1,i]);
                push!(scal_blocks[loc],(lmask,Elt[Elt(1)],Elt(1),Elt[Elt(1)],rmask));

                (lmask,rmask) = masks(indmap_2L[1,i,2,i],indmap_2L[1,i,2,i]);
                push!(scal_blocks[loc],(lmask,Elt[Elt(1)],Elt(1),Elt[Elt(1)],rmask));
                
                (lmask,rmask) = masks(indmap_2L[2,i,2,i],indmap_2L[2,i,2,i]);
                push!(scal_blocks[loc],(lmask,Elt[Elt(1)],Elt(1),Elt[Elt(1)],rmask));
            end
        elseif i > half_basis_size
            (lmask,rmask) = masks(indmap_2R[1,i,1,i],cnt+1);
            push!(op_blocks[i],(lmask,Elt[Elt(1)],bpbp_ut,Elt[Elt(1)],rmask))
            
            (lmask,rmask) = masks(indmap_2R[2,i,1,i],cnt+1);
            push!(op_blocks[i],(lmask,Elt[Elt(1)],bmbp_ut,Elt[Elt(1)],rmask))

            (lmask,rmask) = masks(indmap_2R[1,i,2,i],cnt+1);
            push!(op_blocks[i],(lmask,Elt[Elt(1)],bpbm_ut,Elt[Elt(1)],rmask))

            (lmask,rmask) = masks(indmap_2R[2,i,2,i],cnt+1);
            push!(op_blocks[i],(lmask,Elt[Elt(1)],bmbm_ut,Elt[Elt(1)],rmask))

            for loc in half_basis_size+1:i-1
                (lmask,rmask) = masks(indmap_2R[1,i,1,i],indmap_2R[1,i,1,i]);
                push!(scal_blocks[loc],(lmask,Elt[Elt(1)],Elt(1),Elt[Elt(1)],rmask));

                (lmask,rmask) = masks(indmap_2R[2,i,1,i],indmap_2R[2,i,1,i]);
                push!(scal_blocks[loc],(lmask,Elt[Elt(1)],Elt(1),Elt[Elt(1)],rmask));

                (lmask,rmask) = masks(indmap_2R[1,i,2,i],indmap_2R[1,i,2,i]);
                push!(scal_blocks[loc],(lmask,Elt[Elt(1)],Elt(1),Elt[Elt(1)],rmask));
                
                (lmask,rmask) = masks(indmap_2R[2,i,2,i],indmap_2R[2,i,2,i]);
                push!(scal_blocks[loc],(lmask,Elt[Elt(1)],Elt(1),Elt[Elt(1)],rmask));
            end
        end
    end
    
    # indmap_2 disconnected part
    iso_pp = isomorphism(_lastspace(ap)',_lastspace(ap)');
    iso_mm = isomorphism(_lastspace(am)',_lastspace(am)');
    @plansor p_ai_p[-1 -2;-3 -4] := iso_pp[-1;1]*τ[1 2;-3 -4]*ai[-2;2]
    @plansor m_ai_m[-1 -2;-3 -4] := iso_mm[-1;1]*τ[1 2;-3 -4]*ai[-2;2]
    @plansor p_pm_p[-1 -2;-3 -4] := iso_pp[-1;1]*τ[1 2;-3 -4]*h_pm[-2;2]
    @plansor m_pm_m[-1 -2;-3 -4] := iso_mm[-1;1]*τ[1 2;-3 -4]*h_pm[-2;2]
    
    iso_pppp = pp_f*pp_f';
    iso_pmpm = pm_f*pm_f';
    iso_mmmm = mm_f*mm_f';
    @plansor pp_ai_pp[-1 -2;-3 -4] := iso_pppp[-1;1]*τ[1 2;-3 -4]*ai[-2;2]
    @plansor pm_ai_pm[-1 -2;-3 -4] := iso_pmpm[-1;1]*τ[1 2;-3 -4]*ai[-2;2]
    @plansor mm_ai_mm[-1 -2;-3 -4] := iso_mmmm[-1;1]*τ[1 2;-3 -4]*ai[-2;2]

    @plansor p_ap[-1 -2;-3 -4] := iso_pp[-1;1]*τ[1 2;-3 3]*ap[2 -2;4]*conj(pp_f[-4;3 4]);
    @plansor m_ap[-1 -2;-3 -4] := iso_mm[-1;1]*τ[1 2;-3 3]*ap[2 -2;4]*conj(mp_f[-4;3 4]);
    @plansor p_am[-1 -2;-3 -4] := iso_pp[-1;1]*τ[1 2;-3 3]*am[2 -2;4]*conj(pm_f[-4;3 4]);
    @plansor m_am[-1 -2;-3 -4] := iso_mm[-1;1]*τ[1 2;-3 3]*am[2 -2;4]*conj(mm_f[-4;3 4]);
    @plansor bp_p[-1 -2;-3 -4] := bp[2;-3 3]*iso_mm[1;-4]*τ[4 -2;3 1]*mm_f[-1;2 4]
    @plansor bm_p[-1 -2;-3 -4] := bm[2;-3 3]*iso_mm[1;-4]*τ[4 -2;3 1]*pm_f[-1;2 4]
    @plansor bm_m[-1 -2;-3 -4] := bm[2;-3 3]*iso_pp[1;-4]*τ[4 -2;3 1]*pp_f[-1;2 4]
    @plansor bp_m[-1 -2;-3 -4] := bp[2;-3 3]*iso_pp[1;-4]*τ[4 -2;3 1]*mp_f[-1;2 4]
    
    for i in 1:basis_size, j in i+1:basis_size
        if j < half_basis_size
            (lmask,rmask) = masks(indmap_1L[1,i],indmap_2L[1,i,1,j]);
            push!(op_blocks[j],(lmask,Elt[Elt(1)],p_ap,Elt[Elt(1)],rmask))

            (lmask,rmask) = masks(indmap_1L[1,i],indmap_2L[1,i,2,j]);
            push!(op_blocks[j],(lmask,Elt[Elt(1)],p_am,Elt[Elt(1)],rmask))
            
            (lmask,rmask) = masks(indmap_1L[2,i],indmap_2L[2,i,1,j]);
            push!(op_blocks[j],(lmask,Elt[Elt(1)],m_ap,Elt[Elt(1)],rmask))

            (lmask,rmask) = masks(indmap_1L[2,i],indmap_2L[2,i,2,j]);
            push!(op_blocks[j],(lmask,Elt[Elt(1)],m_am,Elt[Elt(1)],rmask))
            
            for k in j+1:half_basis_size-1
                (lmask,rmask) = masks(indmap_2L[1,i,1,j],indmap_2L[1,i,1,j]);
                push!(scal_blocks[k],(lmask,Elt[Elt(1)],Elt(1),Elt[Elt(1)],rmask));

                (lmask,rmask) = masks(indmap_2L[1,i,2,j],indmap_2L[1,i,2,j]);
                push!(scal_blocks[k],(lmask,Elt[Elt(1)],Elt(1),Elt[Elt(1)],rmask));

                (lmask,rmask) = masks(indmap_2L[2,i,1,j],indmap_2L[2,i,1,j]);
                push!(scal_blocks[k],(lmask,Elt[Elt(1)],Elt(1),Elt[Elt(1)],rmask));

                (lmask,rmask) = masks(indmap_2L[2,i,2,j],indmap_2L[2,i,2,j]);
                push!(scal_blocks[k],(lmask,Elt[Elt(1)],Elt(1),Elt[Elt(1)],rmask));
            end
        end

        if i > half_basis_size
            (lmask,rmask) = masks(indmap_2R[1,i,1,j],indmap_1R[1,j]);
            push!(op_blocks[i],(lmask,Elt[Elt(1)],bp_p,Elt[Elt(1)],rmask))

            (lmask,rmask) = masks(indmap_2R[1,i,2,j],indmap_1R[2,j]);
            push!(op_blocks[i],(lmask,Elt[Elt(1)],bp_m,Elt[Elt(1)],rmask))

            (lmask,rmask) = masks(indmap_2R[2,i,1,j],indmap_1R[1,j]);
            push!(op_blocks[i],(lmask,Elt[Elt(1)],bm_p,Elt[Elt(1)],rmask))
            
            (lmask,rmask) = masks(indmap_2R[2,i,2,j],indmap_1R[2,j]);
            push!(op_blocks[i],(lmask,Elt[Elt(1)],bm_m,Elt[Elt(1)],rmask))

            for k in half_basis_size+1:i-1
                (lmask,rmask) = masks(indmap_2R[1,i,1,j],indmap_2R[1,i,1,j]);
                push!(scal_blocks[k],(lmask,Elt[Elt(1)],Elt(1),Elt[Elt(1)],rmask));

                (lmask,rmask) = masks(indmap_2R[1,i,2,j],indmap_2R[1,i,2,j]);
                push!(scal_blocks[k],(lmask,Elt[Elt(1)],Elt(1),Elt[Elt(1)],rmask));

                (lmask,rmask) = masks(indmap_2R[2,i,1,j],indmap_2R[2,i,1,j]);
                push!(scal_blocks[k],(lmask,Elt[Elt(1)],Elt(1),Elt[Elt(1)],rmask));

                (lmask,rmask) = masks(indmap_2R[2,i,2,j],indmap_2R[2,i,2,j]);
                push!(scal_blocks[k],(lmask,Elt[Elt(1)],Elt(1),Elt[Elt(1)],rmask));
            end
        end
    end

    for loc in 1:half_basis_size, i in 1:half_basis_size,j in i:half_basis_size
        domspaces[loc+1,indmap_2L[1,i,1,j]] = _firstspace(pp_f)
        domspaces[loc+1,indmap_2L[2,i,1,j]] = _firstspace(mp_f)
        domspaces[loc+1,indmap_2L[1,i,2,j]] = _firstspace(pm_f)
        domspaces[loc+1,indmap_2L[2,i,2,j]] = _firstspace(mm_f)
    end
    for loc in half_basis_size+1:basis_size+1, i in half_basis_size+1:basis_size, j in i:basis_size
        domspaces[loc,indmap_2R[1,i,1,j]] = _firstspace(mm_f)
        domspaces[loc,indmap_2R[2,i,1,j]] = _firstspace(pm_f)
        domspaces[loc,indmap_2R[1,i,2,j]] = _firstspace(mp_f)
        domspaces[loc,indmap_2R[2,i,2,j]] = _firstspace(pp_f)
    end

    
    
    #println("offset $(sum(length.(op_blocks)))")
    onsite = fill(add_util_leg(h_pm)*0,basis_size);
    for i in 1:basis_size
        onsite[i] += K[i,i]*add_util_leg(h_pm);
        onsite[i] += V[i,i,i,i]*add_util_leg(h_ppmm);
        onsite[i] += add_util_leg(one(h_pm))*Elt(E0)/basis_size;
    end
    
    for i in 1:half_basis_size-1, j in i+1:half_basis_size
        onsite[i] -= (V[j,i,j,i]+V[i,j,i,j])*add_util_leg(h_pm);
    end
    
    for i in half_basis_size:basis_size, j in i+1:basis_size
        onsite[j] -= (V[i,j,i,j]+V[j,i,j,i])*add_util_leg(h_pm);
    end
    
    for i in 1:half_basis_size-1,j in half_basis_size+1:basis_size
        onsite[j] -=  add_util_leg(h_pm)*(V[i,j,i,j]+V[j,i,j,i])
    end
    
    (lmask,rmask) = masks(1,cnt+1);
    for b in 1:basis_size
        push!(op_blocks[b],(lmask,Elt[Elt(1)],onsite[b],Elt[Elt(1)],rmask));
    end

    #----------------
    
    @plansor ppLm[-1 -2;-3 -4] := bp[-1;1 -2]*h_pm[1;-3]*conj(ut[-4])
    @plansor Lpmm[-1 -2;-3 -4] := bm[-1;-3 1]*h_pm[-2;1]*conj(ut[-4])

    for loc in 2:basis_size
        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); rblocks[end] = Elt(1);
        for i in 1:loc-1
            lblocks[indmap_1L[2,i]] = Elt(K[loc,i])
        end
        push!(op_blocks[loc],block2masks(lblocks,bp_ut,rblocks))

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); rblocks[end] = Elt(1);
        for i in 1:loc-1
            lblocks[indmap_1L[1,i]] = Elt(K[i,loc])
        end
        push!(op_blocks[loc],block2masks(lblocks,bm_ut,rblocks))
    end
    
    
    for loc in 2:basis_size
        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); rblocks[end] = Elt(1);
        for i in 1:loc-1
            lblocks[indmap_1L[2,i]] = Elt(V[loc,loc,i,loc]+V[loc,loc,loc,i])
        end
        push!(op_blocks[loc],block2masks(lblocks,ppLm,rblocks))

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); rblocks[end] = Elt(1);
        for i in 1:loc-1
            lblocks[indmap_1L[1,i]] = Elt(V[loc,i,loc,loc]+V[i,loc,loc,loc])
        end
        push!(op_blocks[loc],block2masks(lblocks,Lpmm,rblocks))
    end

    # fill in all V terms
    #println("onsite $(sum(length.(op_blocks)))")

    # 3|1
    @plansor ppRm[-1 -2;-3 -4] := ut[-1]*ap[1 -2;-4]*h_pm[1;-3]
    @plansor Rpmm[-1 -2;-3 -4] := ut[-1]*h_pm[-2;1]*am[-3 1;-4]
    for i in 1:basis_size-1


        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); lblocks[1] = Elt(1);
        for j in i+1:basis_size
            rblocks[indmap_1R[2,j]] = Elt(V[i,i,j,i]+V[i,i,i,j])
        end
        push!(op_blocks[i],block2masks(lblocks,ppRm,rblocks))
        

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); lblocks[1] = Elt(1);
        for j in i+1:basis_size
            rblocks[indmap_1R[1,j]] = Elt(V[j,i,i,i]+V[i,j,i,i])
        end
        push!(op_blocks[i],block2masks(lblocks,Rpmm,rblocks))
    end
    
    #println("3|1 $(sum(length.(op_blocks)))")

    # 2|2
    # 2|1|1
    # 1|1|2

    # 1|2|1
    @plansor LRmm[-1 -2;-3 -4] := am[1 -2;-4]*bm[-1;-3 1]
    @plansor ppLR[-1 -2;-3 -4] := ap[1 -2;-4]*bp[-1;-3 1]
    @plansor LpRm[-1 -2;-3 -4] := ap[1 -2;-4]*bm[-1;-3 1]
    @plansor RpLm[-1 -2;-3 -4] := bp[-1;1 -2]*am[-3 1;-4]
    @plansor _pm_left[-1 -2;-3 -4] := (mp_f*Lmap_apam_to_pm)[-1]*h_pm[-2;-3]*conj(ut[-4])
    @plansor _pm_right[-1 -2;-3 -4] := ut[-1]*h_pm[-2;-3]*(transpose(Rmap_bpbm_to_pm*pm_f',(1,)))[-4]

    @plansor LRLm_1[-1 -2;-3 -4] := (mp_f_1)[-1;1 2]*bm[2;3 -2]*τ[1 3;-3 -4]
    @plansor LpLR_1[-1 -2;-3 -4] := (mp_f_1)[-1;1 2]*bp[1;-3 3]*τ[3 2;-4 -2]
    @plansor RpLL[-1 -2;-3 -4] := mm_f[-1;1 2]*bp[2;3 -2]*τ[1 3;-3 -4]

    
    @plansor jimm[-1 -2;-3 -4] := iso_pp[-1;1]*ap[-3 2;3]*τ[2 1;4 -2]*conj(pp_f[-4;3 4])
    @plansor ppji[-1 -2;-3 -4] := iso_mm[-1;1]*am[-3 2;3]*τ[2 1;4 -2]*conj(mm_f[-4;3 4])
    @plansor jpim_1[-1 -2;-3 -4] := iso_mm[-1;1]*ap[-3 2;3]*τ[2 1;4 -2]*conj((pm_f_1)[-4;3 4])
    @plansor ipjm_1[-1 -2;-3 -4] := iso_pp[-1;1]*τ[-3 1;2 3]*am[3 -2;4]*conj((pm_f_1)[-4;2 4])


    # 1 2 1
    for i in 1:half_basis_size,j in i+1:half_basis_size-1

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); lblocks[indmap_1L[1,i]] = Elt(1);
        for k in j+1:basis_size
            rblocks[indmap_1R[1,k]] = Elt(V[k,i,j,j]+V[i,k,j,j])
        end
        push!(op_blocks[j],block2masks(lblocks,LRmm,rblocks));

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); lblocks[indmap_1L[2,i]] = Elt(1);
        for k in j+1:basis_size
            rblocks[indmap_1R[2,k]] = Elt(V[j,j,k,i]+V[j,j,i,k])
        end
        push!(op_blocks[j],block2masks(lblocks,ppLR,rblocks));

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); lblocks[indmap_1L[1,i]] = Elt(1);
        for k in j+1:basis_size
            rblocks[indmap_1R[2,k]] = Elt(V[j,i,j,k]+V[i,j,k,j])
        end
        push!(op_blocks[j],block2masks(lblocks,LpRm,rblocks));

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); lblocks[indmap_1L[1,i]] = Elt(1);
        for k in j+1:basis_size
            rblocks[indmap_1R[2,k]] = Elt(V[j,i,k,j]+V[i,j,j,k])
        end
        push!(op_blocks[j],block2masks(lblocks,p_pm_p,rblocks));

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); lblocks[indmap_1L[2,i]] = Elt(1);
        for k in j+1:basis_size
            rblocks[indmap_1R[1,k]] = Elt(V[j,k,j,i]+V[k,j,i,j])
        end
        push!(op_blocks[j],block2masks(lblocks,RpLm,rblocks));

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); lblocks[indmap_1L[2,i]] = Elt(1);
        for k in j+1:basis_size
            rblocks[indmap_1R[1,k]] = Elt(V[j,k,i,j]+V[k,j,j,i])
        end
        push!(op_blocks[j],block2masks(lblocks,m_pm_m,rblocks));
    end
    
    # 2 1 1
    for i in 1:half_basis_size,j in i+1:half_basis_size

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); lblocks[indmap_2L[1,i,1,i]] = Elt(1);
        for k in j+1:basis_size
            rblocks[indmap_1R[2,k]] = Elt(V[i,i,j,k]+V[i,i,k,j])
        end
        push!(op_blocks[j],block2masks(lblocks,bm_m,rblocks));
        
        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); lblocks[indmap_2L[2,i,1,i]] = Elt(1);
        for k in j+1:basis_size
            rblocks[indmap_1R[2,k]] = -2*Elt(V[j,i,i,k]+V[i,j,k,i])
        end
        push!(op_blocks[j],block2masks(lblocks,LpLR_1,rblocks));
        
        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); lblocks[indmap_2L[2,i,1,i]] = Elt(1);
        for k in j+1:basis_size
            rblocks[indmap_1R[2,k]] = Elt(V[i,j,i,k]+V[j,i,k,i])
        end
        push!(op_blocks[j],block2masks(lblocks,bp_m,rblocks));

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); lblocks[indmap_2L[2,i,1,i]] = Elt(1);
        for k in j+1:basis_size
            rblocks[indmap_1R[1,k]] = Elt(V[i,k,i,j]+V[k,i,j,i])*2
            rblocks[indmap_1R[1,k]] -= 2*Elt(V[i,k,j,i]+V[k,i,i,j])
        end
        push!(op_blocks[j],block2masks(lblocks,LRLm_1,rblocks));
        
        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); lblocks[indmap_2L[2,i,1,i]] = Elt(1);
        for k in j+1:basis_size
            rblocks[indmap_1R[1,k]] = -Elt(V[i,k,i,j]+V[k,i,j,i])
        end
        push!(op_blocks[j],block2masks(lblocks, bm_p,rblocks));

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); lblocks[indmap_2L[2,i,2,i]] = Elt(1);
        for k in j+1:basis_size
            rblocks[indmap_1R[1,k]] = Elt(V[j,k,i,i]+V[k,j,i,i])
        end
        push!(op_blocks[j],block2masks(lblocks,RpLL,rblocks));
    end
    
    # 1 2 1
    for j in half_basis_size:basis_size, k in j+1:basis_size
    
        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); rblocks[indmap_1R[1,k]] = Elt(1);
        for i in 1:j-1
            lblocks[indmap_1L[1,i]] = Elt(V[k,i,j,j]+V[i,k,j,j])
        end
        push!(op_blocks[j],block2masks(lblocks,LRmm,rblocks));

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); rblocks[indmap_1R[2,k]] = Elt(1);
        for i in 1:j-1
            lblocks[indmap_1L[2,i]] = Elt(V[j,j,k,i]+V[j,j,i,k])
        end
        push!(op_blocks[j],block2masks(lblocks,ppLR,rblocks));

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); rblocks[indmap_1R[2,k]] = Elt(1);
        for i in 1:j-1
            lblocks[indmap_1L[1,i]] = Elt(V[j,i,j,k]+V[i,j,k,j])
        end
        push!(op_blocks[j],block2masks(lblocks,LpRm,rblocks));

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); rblocks[indmap_1R[2,k]] = Elt(1);
        for i in 1:j-1
            lblocks[indmap_1L[1,i]] = Elt(V[j,i,k,j]+V[i,j,j,k])
        end
        push!(op_blocks[j],block2masks(lblocks,p_pm_p,rblocks));

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); rblocks[indmap_1R[1,k]] = Elt(1);
        for i in 1:j-1
            lblocks[indmap_1L[2,i]] = Elt(V[j,k,j,i]+V[k,j,i,j])
        end
        push!(op_blocks[j],block2masks(lblocks,RpLm,rblocks));

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); rblocks[indmap_1R[1,k]] = Elt(1);
        for i in 1:j-1
            lblocks[indmap_1L[2,i]] = Elt(V[j,k,i,j]+V[k,j,j,i])
        end
        push!(op_blocks[j],block2masks(lblocks,m_pm_m,rblocks));
    end 
    
    # 1 1 2
    for j in half_basis_size:basis_size, k in j+1:basis_size
        lmask = fill(false,cnt+1); rmask = fill(false,cnt+1);
        rblock = Vector{Elt}(undef,cnt+1); rmask[indmap_2R[2,k,2,k]] = true;
        for i in 1:j-1
            lmask[indmap_1L[1,i]] = true;
            rblock[indmap_1L[1,i]] = Elt(V[j,i,k,k]+V[i,j,k,k])
        end
        push!(op_blocks[j],(lmask,rblock[lmask],jimm,Elt[Elt(1)],rmask));

        lmask = fill(false,cnt+1); rmask = fill(false,cnt+1);
        rblock = Vector{Elt}(undef,cnt+1); rmask[indmap_2R[1,k,1,k]] = true;
        for i in 1:j-1
            lmask[indmap_1L[2,i]] = true;
            rblock[indmap_1L[2,i]] = Elt(V[k,k,j,i]+V[k,k,i,j])
        end
        push!(op_blocks[j],(lmask,rblock[lmask],ppji,Elt[Elt(1)],rmask));

        lmask = fill(false,cnt+1); rmask = fill(false,cnt+1);
        rblock = Vector{Elt}(undef,cnt+1); rmask[indmap_2R[2,k,1,k]] = true;
        for i in 1:j-1
            
            lmask[indmap_1L[2,i]] = true;
            rblock[indmap_1L[2,i]] = Elt(V[j,k,i,k]+V[k,j,k,i])
            rblock[indmap_1L[2,i]] -= 2*Elt(V[k,j,i,k]+V[j,k,k,i])
        end
        push!(op_blocks[j],(lmask,rblock[lmask],jpim_1,Elt[Elt(1)],rmask));
        
        lmask = fill(false,cnt+1); rmask = fill(false,cnt+1);
        rblock = Vector{Elt}(undef,cnt+1); rmask[indmap_2R[2,k,1,k]] = true;
        for i in 1:j-1
            
            lmask[indmap_1L[2,i]] = true;
            rblock[indmap_1L[2,i]] = Elt(V[j,k,i,k]+V[k,j,k,i])
        end
        push!(op_blocks[j],(lmask,rblock[lmask],jpim_1 - m_ap,Elt[Elt(1)],rmask));
        
        
        lmask = fill(false,cnt+1); rmask = fill(false,cnt+1);
        rblock = Vector{Elt}(undef,cnt+1); rmask[indmap_2R[2,k,1,k]] = true;
        for i in 1:j-1
            
            lmask[indmap_1L[1,i]] = true;
            rblock[indmap_1L[1,i]] = Elt(V[i,k,j,k]+V[k,i,k,j])
            rblock[indmap_1L[1,i]] -= 2*Elt(V[i,k,k,j]+V[k,i,j,k])
        end
        push!(op_blocks[j],(lmask,rblock[lmask],ipjm_1,Elt[Elt(1)],rmask));
        
        lmask = fill(false,cnt+1); rmask = fill(false,cnt+1);
        rblock = Vector{Elt}(undef,cnt+1); rmask[indmap_2R[2,k,1,k]] = true;
        for i in 1:j-1
            
            lmask[indmap_1L[1,i]] = true;
            rblock[indmap_1L[1,i]] = Elt(V[i,k,j,k]+V[k,i,k,j])
        end
        push!(op_blocks[j],(lmask,rblock[lmask],p_am - ipjm_1,Elt[Elt(1)],rmask));    
    end

    # 1 1 2 + 2 2
    for k in 2:half_basis_size
        lmask = fill(false,cnt+1); rmask = fill(false,cnt+1); rmask[end] = true;
        lblocks = Vector{Elt}(undef,cnt+1);
        for i in 1:min(half_basis_size-1,k-1)
            lmask[indmap_2L[1,i,1,i]] = true;
            lblocks[indmap_2L[1,i,1,i]] = Elt(V[i,i,k,k])
        end
        for i in 1:k-1,j in i+1:k-1
            lmask[indmap_2L[1,i,1,j]] = true;
            lblocks[indmap_2L[1,i,1,j]] = Elt(V[i,j,k,k]+V[j,i,k,k])
        end
        push!(op_blocks[k],(lmask,lblocks[lmask],bmbm_ut,Elt[Elt(1)],rmask))


        lmask = fill(false,cnt+1); rmask = fill(false,cnt+1); rmask[end] = true;
        lblocks = Vector{Elt}(undef,cnt+1);
        for i in 1:min(half_basis_size-1,k-1)
            lmask[indmap_2L[2,i,2,i]] = true;
            lblocks[indmap_2L[2,i,2,i]] = Elt(V[k,k,i,i])
        end
        for i in 1:k-1,j in i+1:k-1
            lmask[indmap_2L[2,i,2,j]] = true;
            lblocks[indmap_2L[2,i,2,j]] = Elt(V[k,k,j,i]+V[k,k,i,j])
        end
        push!(op_blocks[k],(lmask,lblocks[lmask],bpbp_ut,Elt[Elt(1)],rmask))


        lmask = fill(false,cnt+1); rmask = fill(false,cnt+1); rmask[end] = true;
        lblocks = Vector{Elt}(undef,cnt+1);
        for i in 1:min(half_basis_size-1,k-1)
            lmask[indmap_2L[2,i,1,i]] = true;
            lblocks[indmap_2L[2,i,1,i]] = Elt(V[k,i,k,i]+V[i,k,i,k])
        end
        push!(op_blocks[k],(lmask,lblocks[lmask],bpbm_ut,Elt[Elt(1)],rmask))

        lmask = fill(false,cnt+1); rmask = fill(false,cnt+1); rmask[end] = true;
        lblocks = Vector{Elt}(undef,cnt+1);
        for i in 1:min(half_basis_size-1,k-1)
            lmask[indmap_2L[2,i,1,i]] = true;
            lblocks[indmap_2L[2,i,1,i]] = Elt(V[k,i,i,k]+V[i,k,k,i])
        end
        for i in 1:k-1,j in i+1:k-1
            lmask[indmap_2L[1,i,2,j]] = true;
            lmask[indmap_2L[2,i,1,j]] = true;

            lblocks[indmap_2L[2,i,1,j]] = -Elt(V[k,j,k,i]+V[j,k,i,k]);

            lblocks[indmap_2L[2,i,1,j]] += Elt(V[j,k,k,i]+V[k,j,i,k]);
            lblocks[indmap_2L[1,i,2,j]] = Elt(V[i,k,k,j]+V[k,i,j,k]);
        end
        push!(op_blocks[k],(lmask,lblocks[lmask],_pm_left,Elt[Elt(1)],rmask))

        lmask = fill(false,cnt+1); rmask = fill(false,cnt+1); rmask[end] = true;
        lblock = Vector{Elt}(undef,cnt+1);
        for i in 1:k-1,j in i+1:k-1
            lmask[indmap_2L[1,i,2,j]] = true;
            lmask[indmap_2L[2,i,1,j]] = true;

            lblock[indmap_2L[2,i,1,j]] = -Elt(V[k,j,k,i]+V[j,k,i,k]);
            lblock[indmap_2L[1,i,2,j]] = Elt(V[k,i,k,j]+V[i,k,j,k]);

        end
        push!(op_blocks[k],(lmask,lblock[lmask],bmbp_ut ,Elt[Elt(1)],rmask));

    end
    
    # 2 1 1 + 2 2
    for i in half_basis_size:basis_size
        lmask = fill(false,cnt+1); rmask = fill(false,cnt+1); lmask[1] = true;
        rblocks = Vector{Elt}(undef,cnt+1);
        for j in i+1:basis_size
            rmask[indmap_2R[1,j,1,j]] = true;
            rblocks[indmap_2R[1,j,1,j]] = Elt(V[j,j,i,i])
        end
        for j in i+1:basis_size,k in j+1:basis_size
            rmask[indmap_2R[1,j,1,k]] = true;
            rblocks[indmap_2R[1,j,1,k]] = Elt(V[j,k,i,i]+V[k,j,i,i]);
        end
        push!(op_blocks[i],(lmask,Elt[Elt(1)],ut_amam,rblocks[rmask],rmask))
        

        lmask = fill(false,cnt+1); rmask = fill(false,cnt+1); lmask[1] = true;
        rblocks = Vector{Elt}(undef,cnt+1);
        for j in i+1:basis_size
            rmask[indmap_2R[2,j,2,j]] = true;
            rblocks[indmap_2R[2,j,2,j]] = Elt(V[i,i,j,j])
        end
        for j in i+1:basis_size,k in j+1:basis_size
            rmask[indmap_2R[2,j,2,k]] = true;
            rblocks[indmap_2R[2,j,2,k]] = Elt(V[i,i,j,k]+V[i,i,k,j]);
        end
        push!(op_blocks[i],(lmask,Elt[Elt(1)],ut_apap,rblocks[rmask],rmask))

        lmask = fill(false,cnt+1); rmask = fill(false,cnt+1); lmask[1] = true;
        rblocks = Vector{Elt}(undef,cnt+1);
        for j in i+1:basis_size
            rmask[indmap_2R[2,j,1,j]] = true;
            rblocks[indmap_2R[2,j,1,j]] = Elt(V[i,j,i,j]+V[j,i,j,i])
        end
        push!(op_blocks[i],(lmask,Elt[Elt(1)],ut_apam,rblocks[rmask],rmask))

        lmask = fill(false,cnt+1); rmask = fill(false,cnt+1); lmask[1] = true;
        rblocks = Vector{Elt}(undef,cnt+1);
        for j in i+1:basis_size,k in j+1:basis_size            
            rmask[indmap_2R[1,j,2,k]] = true;
            rmask[indmap_2R[2,j,1,k]] = true;
            rblocks[indmap_2R[1,j,2,k]] = Elt(V[i,j,k,i]+V[j,i,i,k]);

            rblocks[indmap_2R[2,j,1,k]] = -Elt(V[i,k,i,j]+V[k,i,j,i]);
            rblocks[indmap_2R[2,j,1,k]] += Elt(V[k,i,i,j]+V[i,k,j,i]);
        end
        for j in i+1:basis_size
            rmask[indmap_2R[2,j,1,j]] = true;
            rblocks[indmap_2R[2,j,1,j]] = Elt(V[j,i,i,j]+V[i,j,j,i])
        end
        push!(op_blocks[i],(lmask,Elt[Elt(1)],_pm_right,rblocks[rmask],rmask))


        lmask = fill(false,cnt+1); rmask = fill(false,cnt+1); lmask[1] = true;
        rblocks = Vector{Elt}(undef,cnt+1);
        for j in i+1:basis_size,k in j+1:basis_size
            rmask[indmap_2R[1,j,2,k]] = true;
            rmask[indmap_2R[2,j,1,k]] = true;
            rblocks[indmap_2R[1,j,2,k]] = Elt(V[j,i,k,i]+V[i,j,i,k]);
            rblocks[indmap_2R[2,j,1,k]] = Elt(V[i,k,i,j]+V[k,i,j,i])/(-1);
        end
        push!(op_blocks[i],(lmask,Elt[Elt(1)],ut_amap,rblocks[rmask],rmask))
    end
    
    #println("2|2 $(sum(length.(op_blocks)))")

    
    # 1|1|1|1
    # (i,j) in indmap_2, 1 in indmap_4, 1 onsite

    @plansor jimR_1[-1 -2;-3 -4] := pp_f_1[-1;1 2]*τ[3 2;-4 -2]*bm[1;-3 3]
    
    
    # 3 left of half_basis_size
    for k in 3:half_basis_size,l in k+1:basis_size
        numblocks = 0
        for a in 1:k-1, b in a+1:k-1
            numblocks += 1
        end
        numblocks > basis_size-k || continue
        numblocks == 0 && continue

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); rblocks[indmap_1R[2,l]] = Elt(1)
        for i in 1:k-1, j in i+1:k-1
            lblocks[indmap_2L[2,i,1,j]] = (V[j,k,l,i]+V[k,j,i,l])*(-2);

            lblocks[indmap_2L[1,i,2,j]] = (V[i,k,l,j]+V[k,i,j,l])*(-2);
            lblocks[indmap_2L[1,i,2,j]] += (V[k,i,l,j]+V[i,k,j,l])*2;
        end
        push!(op_blocks[k],block2masks(lblocks,LpLR_1,rblocks));

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); rblocks[indmap_1R[2,l]] = Elt(1);
        for i in 1:k-1, j in i+1:k-1
            lblocks[indmap_2L[2,i,1,j]] = (V[k,j,l,i]+V[j,k,i,l]);
            lblocks[indmap_2L[1,i,2,j]] = -(V[k,i,l,j]+V[i,k,j,l]);
        end
        push!(op_blocks[k],block2masks(lblocks,bp_m ,rblocks));


        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); rblocks[indmap_1R[1,l]] = Elt(1);
        for i in 1:k-1, j in i+1:k-1
            lblocks[indmap_2L[2,i,1,j]] = (V[j,l,k,i]+V[l,j,i,k])*(-2);
            lblocks[indmap_2L[2,i,1,j]] += (V[l,j,k,i]+V[j,l,i,k])*2;

            lblocks[indmap_2L[1,i,2,j]] = (V[i,l,k,j]+V[l,i,j,k])*(-2);
        end
        push!(op_blocks[k],block2masks(lblocks, LRLm_1,rblocks));

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); rblocks[indmap_1R[1,l]] = Elt(1);
        for i in 1:k-1, j in i+1:k-1
            lblocks[indmap_2L[2,i,1,j]] = -(V[l,j,k,i]+V[j,l,i,k]);
            lblocks[indmap_2L[1,i,2,j]] = (V[l,i,k,j]+V[i,l,j,k]);
        end
        push!(op_blocks[k],block2masks(lblocks, bm_p,rblocks));


        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); rblocks[indmap_1R[2,l]] = Elt(1);
        for i in 1:k-1, j in i+1:k-1
            lblocks[indmap_2L[1,i,1,j]] = 2*(V[j,i,l,k]+V[i,j,k,l]);
        end
        push!(op_blocks[k],block2masks(lblocks,jimR_1,rblocks));
        
        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); rblocks[indmap_1R[2,l]] = Elt(1);
        for i in 1:k-1, j in i+1:k-1
            lblocks[indmap_2L[1,i,1,j]] = (V[i,j,l,k]+V[j,i,k,l]);
            lblocks[indmap_2L[1,i,1,j]] -= (V[j,i,l,k]+V[i,j,k,l]);
        end
        push!(op_blocks[k],block2masks(lblocks,bm_m,rblocks));

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); rblocks[indmap_1R[1,l]] = Elt(1);
        for i in 1:k-1, j in i+1:k-1
            lblocks[indmap_2L[2,i,2,j]] = (V[l,k,j,i]+V[k,l,i,j]);
        end
        push!(op_blocks[k],block2masks(lblocks,RpLL,rblocks));

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); rblocks[indmap_1R[1,l]] = Elt(1);
        for i in 1:k-1, j in i+1:k-1
            lblocks[indmap_2L[2,i,2,j]] = (V[l,k,i,j]+V[k,l,j,i]);
        end
        push!(op_blocks[k],block2masks(lblocks,bp_p,rblocks));
    end
    #println("1|1|1|1 (1,3) $(sum(length.(op_blocks)))")
    
    for k in 3:half_basis_size, i in 1:k-1, j in i+1:k-1
        numblocks = 0
        for a in 1:k-1, b in a+1:k-1
            numblocks += 1
        end
        numblocks <= basis_size-k || continue
        numblocks == 0 && continue

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); lblocks[indmap_2L[2,i,1,j]] = Elt(1)
        for l in k+1:basis_size
            rblocks[indmap_1R[2,l]] = (V[j,k,l,i]+V[k,j,i,l])*(-2);
        end
        push!(op_blocks[k],block2masks(lblocks,LpLR_1,rblocks));

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); lblocks[indmap_2L[2,i,1,j]] = Elt(1);
        for l in k+1:basis_size
            rblocks[indmap_1R[2,l]] = (V[k,j,l,i]+V[j,k,i,l]);
        end
        push!(op_blocks[k],block2masks(lblocks,bp_m,rblocks));

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); lblocks[indmap_2L[1,i,2,j]] = Elt(1);
        for l in k+1:basis_size
            rblocks[indmap_1R[2,l]] = -(V[k,i,l,j]+V[i,k,j,l]);
        end
        push!(op_blocks[k],block2masks(lblocks,bp_m,rblocks));

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); lblocks[indmap_2L[1,i,2,j]] = Elt(1)
        for l in k+1:basis_size
            rblocks[indmap_1R[2,l]] = (V[i,k,l,j]+V[k,i,j,l])*(-2);
            rblocks[indmap_1R[2,l]] += (V[k,i,l,j]+V[i,k,j,l])*2;
        end
        push!(op_blocks[k],block2masks(lblocks,LpLR_1,rblocks));


        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); lblocks[indmap_2L[2,i,1,j]] = Elt(1);
        for l in k+1:basis_size
            rblocks[indmap_1R[1,l]] = (V[j,l,k,i]+V[l,j,i,k])*(-2);
            rblocks[indmap_1R[1,l]] += (V[l,j,k,i]+V[j,l,i,k])*2;
        end
        push!(op_blocks[k],block2masks(lblocks,LRLm_1,rblocks));


        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); lblocks[indmap_2L[2,i,1,j]] = Elt(1);
        for l in k+1:basis_size
            rblocks[indmap_1R[1,l]] = -(V[l,j,k,i]+V[j,l,i,k]);
        end
        push!(op_blocks[k],block2masks(lblocks, bm_p,rblocks));


        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); lblocks[indmap_2L[1,i,2,j]] = Elt(1);
        for l in k+1:basis_size
            rblocks[indmap_1R[1,l]] = (V[i,l,k,j]+V[l,i,j,k])*(-2);
        end
        push!(op_blocks[k],block2masks(lblocks,LRLm_1,rblocks));


        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1);lblocks[indmap_2L[1,i,2,j]] = Elt(1);
        for l in k+1:basis_size
            rblocks[indmap_1R[1,l]] = (V[l,i,k,j]+V[i,l,j,k]);
        end
        push!(op_blocks[k],block2masks(lblocks, bm_p,rblocks));

        
        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); lblocks[indmap_2L[1,i,1,j]] = Elt(1);
        for l in k+1:basis_size            
            rblocks[indmap_1R[2,l]]  += (V[j,i,l,k]+V[i,j,k,l])*2;
        end
        push!(op_blocks[k],block2masks(lblocks,jimR_1,rblocks));
        
        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); lblocks[indmap_2L[1,i,1,j]] = Elt(1);
        for l in k+1:basis_size
            rblocks[indmap_1R[2,l]] = (V[i,j,l,k]+V[j,i,k,l]);
            rblocks[indmap_1R[2,l]] -= (V[j,i,l,k]+V[i,j,k,l]);
        end
        push!(op_blocks[k],block2masks(lblocks,bm_m,rblocks));
        
        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); lblocks[indmap_2L[2,i,2,j]] = Elt(1);
        for l in k+1:basis_size
            rblocks[indmap_1R[1,l]] = (V[l,k,j,i]+V[k,l,i,j]);
        end
        push!(op_blocks[k],block2masks(lblocks,RpLL,rblocks));
        
        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); lblocks[indmap_2L[2,i,2,j]] = Elt(1);
        for l in k+1:basis_size
            rblocks[indmap_1R[1,l]] = (V[l,k,i,j]+V[k,l,j,i]);
           
        end
        push!(op_blocks[k],block2masks(lblocks,bp_p,rblocks));
        
    end
    #println("1|1|1|1 (1,3) $(sum(length.(op_blocks)))")
    
    # 3 right of half_basis_size
    for i in 1:basis_size,j in i+1:basis_size
        j >= half_basis_size || continue
        
        numblocks = 0
        for a in max(j+1,half_basis_size+1):basis_size, b in a+1:basis_size
            numblocks += 1
        end
        numblocks/8 > (j-1)/12 || continue
        numblocks == 0 && continue

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); lblocks[indmap_1L[2,i]] = Elt(1);
        for k in max(j+1,half_basis_size+1):basis_size,l in k+1:basis_size
            rblocks[indmap_2R[1,k,2,l]] = (V[j,k,i,l]+V[k,j,l,i])/(-2)
            rblocks[indmap_2R[1,k,2,l]] += (V[k,j,i,l]+V[j,k,l,i])

            rblocks[indmap_2R[2,k,1,l]] = (V[l,j,k,i]+V[j,l,i,k])/(-2)
            rblocks[indmap_2R[2,k,1,l]] += (V[l,j,i,k]+V[j,l,k,i]);
        end
        push!(op_blocks[j],block2masks(lblocks,-2*jpim_1,rblocks));
        
        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); lblocks[indmap_1L[2,i]] = Elt(1);
        for k in max(j+1,half_basis_size+1):basis_size,l in k+1:basis_size
            rblocks[indmap_2R[1,k,2,l]] = (V[j,k,i,l]+V[k,j,l,i])
            rblocks[indmap_2R[2,k,1,l]] = -(V[l,j,k,i]+V[j,l,i,k])
        end
        push!(op_blocks[j],block2masks(lblocks,m_ap-jpim_1,rblocks));
 
        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); lblocks[indmap_1L[1,i]] = Elt(1);
        for k in max(j+1,half_basis_size+1):basis_size,l in k+1:basis_size
            rblocks[indmap_2R[1,k,2,l]] = (V[i,k,j,l]+V[k,i,l,j])/(-2)
            rblocks[indmap_2R[1,k,2,l]] += (V[k,i,j,l]+V[i,k,l,j])

            rblocks[indmap_2R[2,k,1,l]] = (V[l,i,k,j]+V[i,l,j,k])/(-2)
            rblocks[indmap_2R[2,k,1,l]] += (V[l,i,j,k]+V[i,l,k,j]);
        end
        push!(op_blocks[j],block2masks(lblocks,-2*ipjm_1,rblocks));

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); lblocks[indmap_1L[1,i]] = Elt(1);
        for k in max(j+1,half_basis_size+1):basis_size,l in k+1:basis_size
            rblocks[indmap_2R[1,k,2,l]] = (V[i,k,j,l]+V[k,i,l,j])
            rblocks[indmap_2R[2,k,1,l]] = -(V[l,i,k,j]+V[i,l,j,k])
        end
        push!(op_blocks[j],block2masks(lblocks,-(p_am-ipjm_1),rblocks));
    
        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); lblocks[indmap_1L[2,i]] = Elt(1);
        for k in max(j+1,half_basis_size+1):basis_size,l in k+1:basis_size
            rblocks[indmap_2R[1,k,1,l]] = (V[l,k,j,i]+V[k,l,i,j])
            rblocks[indmap_2R[1,k,1,l]] += (V[l,k,i,j]+V[k,l,j,i])
        end
        push!(op_blocks[j],block2masks(lblocks,(m_am + ppji)/2,rblocks));

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); lblocks[indmap_1L[2,i]] = Elt(1);
        for k in max(j+1,half_basis_size+1):basis_size,l in k+1:basis_size
            rblocks[indmap_2R[1,k,1,l]] = -(V[l,k,j,i]+V[k,l,i,j])
            rblocks[indmap_2R[1,k,1,l]] += (V[l,k,i,j]+V[k,l,j,i])
        end
        push!(op_blocks[j],block2masks(lblocks,(m_am - ppji)/2,rblocks));

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); lblocks[indmap_1L[1,i]] = Elt(1);
        for k in max(j+1,half_basis_size+1):basis_size,l in k+1:basis_size
            rblocks[indmap_2R[2,k,2,l]] = (V[j,i,k,l]+V[i,j,l,k])
            rblocks[indmap_2R[2,k,2,l]] += (V[i,j,k,l]+V[j,i,l,k])
        end
        push!(op_blocks[j],block2masks(lblocks,(p_ap+jimm)/2,rblocks));

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); lblocks[indmap_1L[1,i]] = Elt(1);
        for k in max(j+1,half_basis_size+1):basis_size,l in k+1:basis_size
            rblocks[indmap_2R[2,k,2,l]] = (V[j,i,k,l]+V[i,j,l,k])
            rblocks[indmap_2R[2,k,2,l]] -= (V[i,j,k,l]+V[j,i,l,k])
        end
        push!(op_blocks[j],block2masks(lblocks,(p_ap-jimm)/2,rblocks));
        
    end
    
    #println("1|1|1|1 (3,1) $(sum(length.(op_blocks)))") 
    for j in half_basis_size:basis_size,k in max(j+1,half_basis_size+1):basis_size,l in k+1:basis_size
        j >= half_basis_size || continue
        
        numblocks = 0
        for a in max(j+1,half_basis_size+1):basis_size, b in a+1:basis_size
            numblocks += 1
        end
        numblocks/8 <= (j-1)/12 || continue
        numblocks == 0 && continue

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); rblocks[indmap_2R[1,k,2,l]] = Elt(1);
        for i in 1:j-1
            lblocks[indmap_1L[2,i]] = (V[j,k,i,l]+V[k,j,l,i])/(-2)
            lblocks[indmap_1L[2,i]] += (V[k,j,i,l]+V[j,k,l,i])
        end
        push!(op_blocks[j],block2masks(lblocks,-2*jpim_1,rblocks));

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); rblocks[indmap_2R[2,k,1,l]] = Elt(1);
        for i in 1:j-1
            lblocks[indmap_1L[2,i]] = (V[l,j,k,i]+V[j,l,i,k])/(-2)
            lblocks[indmap_1L[2,i]] += (V[l,j,i,k]+V[j,l,k,i]);
        end
        push!(op_blocks[j],block2masks(lblocks,-2*jpim_1,rblocks));
        
        
        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); rblocks[indmap_2R[1,k,2,l]] = Elt(1);
        for i in 1:j-1
            lblocks[indmap_1L[2,i]] = (V[j,k,i,l]+V[k,j,l,i])
        end
        push!(op_blocks[j],block2masks(lblocks,m_ap-jpim_1,rblocks));
        
        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); rblocks[indmap_2R[2,k,1,l]] = Elt(1);
        for i in 1:j-1
            lblocks[indmap_1L[2,i]] = -(V[l,j,k,i]+V[j,l,i,k])
        end
        push!(op_blocks[j],block2masks(lblocks,m_ap-jpim_1,rblocks));
 
        
        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); rblocks[indmap_2R[1,k,2,l]] = Elt(1);
        for i in 1:j-1
            lblocks[indmap_1L[1,i]] = (V[i,k,j,l]+V[k,i,l,j])/(-2)
            lblocks[indmap_1L[1,i]] += (V[k,i,j,l]+V[i,k,l,j])
        end
        push!(op_blocks[j],block2masks(lblocks,-2*ipjm_1,rblocks));

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); rblocks[indmap_2R[2,k,1,l]] = Elt(1);
        for i in 1:j-1
            lblocks[indmap_1L[1,i]] = (V[l,i,k,j]+V[i,l,j,k])/(-2)
            lblocks[indmap_1L[1,i]] += (V[l,i,j,k]+V[i,l,k,j]);
        end
        push!(op_blocks[j],block2masks(lblocks,-2*ipjm_1,rblocks));

        
        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); rblocks[indmap_2R[1,k,2,l]] = Elt(1);
        for i in 1:j-1
            lblocks[indmap_1L[1,i]] = (V[i,k,j,l]+V[k,i,l,j])
        end
        push!(op_blocks[j],block2masks(lblocks,-(p_am-ipjm_1),rblocks));

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); rblocks[indmap_2R[2,k,1,l]] = Elt(1);
        for i in 1:j-1
            lblocks[indmap_1L[1,i]] = -(V[l,i,k,j]+V[i,l,j,k])
        end
        push!(op_blocks[j],block2masks(lblocks,-(p_am-ipjm_1),rblocks));
    
        
        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); rblocks[indmap_2R[1,k,1,l]] = Elt(1);
        for i in 1:j-1
            lblocks[indmap_1L[2,i]] = (V[l,k,j,i]+V[k,l,i,j])
            lblocks[indmap_1L[2,i]] += (V[l,k,i,j]+V[k,l,j,i])
        end
        push!(op_blocks[j],block2masks(lblocks,(m_am + ppji)/2,rblocks));

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); rblocks[indmap_2R[1,k,1,l]] = Elt(1);
        for i in 1:j-1
            lblocks[indmap_1L[2,i]] = -(V[l,k,j,i]+V[k,l,i,j])
            lblocks[indmap_1L[2,i]] += (V[l,k,i,j]+V[k,l,j,i])
        end
        push!(op_blocks[j],block2masks(lblocks,(m_am - ppji)/2,rblocks));

        
        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); rblocks[indmap_2R[2,k,2,l]]  = Elt(1);
        for i in 1:j-1
            lblocks[indmap_1L[1,i]] = (V[j,i,k,l]+V[i,j,l,k])
            lblocks[indmap_1L[1,i]] += (V[i,j,k,l]+V[j,i,l,k])
        end
        push!(op_blocks[j],block2masks(lblocks,(p_ap+jimm)/2,rblocks));

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); rblocks[indmap_2R[2,k,2,l]]  = Elt(1);
        for i in 1:j-1
            lblocks[indmap_1L[1,i]] = (V[j,i,k,l]+V[i,j,l,k])
            lblocks[indmap_1L[1,i]] -= (V[i,j,k,l]+V[j,i,l,k])
        end
        push!(op_blocks[j],block2masks(lblocks,(p_ap-jimm)/2,rblocks));
        
    end
    #println("1|1|1|1 (3,1) $(sum(length.(op_blocks)))")

    # loc == half_basis_size:
    @plansor jkil_2[-1 -2;-3 -4] := mp_f_2[-1;1 2]*ai[3;-3]*τ[3 1;4 5]*τ[5 2;6 -2]*conj(mp_f_2[-4;4 6])
    @plansor jikl_1[-1 -2;-3 -4] := pp_f_1[-1;1 2]*ai[3;-3]*τ[3 1;4 5]*τ[5 2;6 -2]*conj(pp_f_1[-4;4 6])
    @plansor lkij_1[-1 -2;-3 -4] := mm_f_1[-1;1 2]*ai[3;-3]*τ[3 1;4 5]*τ[5 2;6 -2]*conj(mm_f_1[-4;4 6])

    
    
    for k in half_basis_size+1:basis_size, l in k+1:basis_size

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); rblocks[indmap_2R[1,k,2,l]] = Elt(1);
        for i in 1:half_basis_size-1,j in i+1:half_basis_size-1
            lblocks[indmap_2L[2,i,1,j]] = Elt(V[j,k,i,l]+V[k,j,l,i])
            lblocks[indmap_2L[2,i,1,j]] += Elt(V[k,j,i,l]+V[j,k,l,i])*(-2)
            lblocks[indmap_2L[1,i,2,j]] = Elt(V[k,i,j,l]+V[i,k,l,j])*(-2)
            lblocks[indmap_2L[1,i,2,j]] += Elt(V[i,k,j,l]+V[k,i,l,j])
        end
        push!(scal_blocks[half_basis_size],block2masks(lblocks,Elt(1),rblocks));


        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); rblocks[indmap_2R[2,k,1,l]] = Elt(1);
        for i in 1:half_basis_size-1,j in i+1:half_basis_size-1
            lblocks[indmap_2L[1,i,2,j]] = Elt(V[l,i,k,j]+V[i,l,j,k])
            lblocks[indmap_2L[1,i,2,j]] += Elt(V[l,i,j,k]+V[i,l,k,j])*(-2)

            lblocks[indmap_2L[2,i,1,j]] = Elt(V[l,j,k,i]+V[j,l,i,k])
            lblocks[indmap_2L[2,i,1,j]] += Elt(V[l,j,i,k]+V[j,l,k,i])*(-2)
        end
        push!(scal_blocks[half_basis_size],block2masks(lblocks,Elt(1),rblocks));        

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); rblocks[indmap_2R[2,k,2,l]] = Elt(1);
        for i in 1:half_basis_size-1,j in i+1:half_basis_size-1
            lblocks[indmap_2L[1,i,1,j]] = Elt(V[j,i,k,l]+V[i,j,l,k])
            lblocks[indmap_2L[1,i,1,j]] -= Elt(V[i,j,k,l]+V[j,i,l,k])
        end
        push!(scal_blocks[half_basis_size],block2masks(lblocks,Elt(1),rblocks));        
        
        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); rblocks[indmap_2R[1,k,1,l]] = Elt(1);
        for i in 1:half_basis_size-1,j in i+1:half_basis_size-1
            lblocks[indmap_2L[2,i,2,j]] = Elt(V[l,k,i,j]+V[k,l,j,i])
            lblocks[indmap_2L[2,i,2,j]] -= Elt(V[l,k,j,i]+V[k,l,i,j])
        end
        push!(scal_blocks[half_basis_size],block2masks(lblocks,Elt(1),rblocks));        
        

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); rblocks[indmap_2R[1,k,2,l]] = Elt(1);
        for i in 1:half_basis_size-1,j in i+1:half_basis_size-1
            lblocks[indmap_2L[2,i,1,j]] = Elt(V[k,j,i,l]+V[j,k,l,i])*2
            lblocks[indmap_2L[1,i,2,j]] = Elt(V[k,i,j,l]+V[i,k,l,j])*2
            lblocks[indmap_2L[1,i,2,j]] += Elt(V[i,k,j,l]+V[k,i,l,j])*(-2)
        end
        push!(op_blocks[half_basis_size],block2masks(lblocks,jkil_2,rblocks));


        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); rblocks[indmap_2R[2,k,1,l]] = Elt(1);
        for i in 1:half_basis_size-1,j in i+1:half_basis_size-1
            lblocks[indmap_2L[1,i,2,j]] = Elt(V[l,i,j,k]+V[i,l,k,j])*2
            lblocks[indmap_2L[2,i,1,j]] = Elt(V[l,j,k,i]+V[j,l,i,k])*(-2)
            lblocks[indmap_2L[2,i,1,j]] += Elt(V[l,j,i,k]+V[j,l,k,i])*2
        end
        push!(op_blocks[half_basis_size],block2masks(lblocks,jkil_2,rblocks));
        
        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); rblocks[indmap_2R[2,k,2,l]] = Elt(1);
        for i in 1:half_basis_size-1,j in i+1:half_basis_size-1
            lblocks[indmap_2L[1,i,1,j]] = Elt(V[i,j,k,l]+V[j,i,l,k])*2
        end
        push!(op_blocks[half_basis_size],block2masks(lblocks,jikl_1,rblocks));

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); rblocks[indmap_2R[1,k,1,l]] = Elt(1);
        for i in 1:half_basis_size-1,j in i+1:half_basis_size-1
            lblocks[indmap_2L[2,i,2,j]] = Elt(V[l,k,j,i]+V[k,l,i,j])*2
        end
        push!(op_blocks[half_basis_size],block2masks(lblocks,lkij_1,rblocks));
    end

    for k in half_basis_size+1:basis_size

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); rblocks[indmap_2R[2,k,1,k]] = Elt(1);
        for i in 1:half_basis_size-1
            lblocks[indmap_2L[1,i,2,i]] = Elt(V[i,k,i,k]+V[k,i,k,i])
            lblocks[indmap_2L[2,i,1,i]] = Elt(V[i,k,k,i]+V[k,i,i,k])*(-2)
        end
        for i in 1:half_basis_size-1, j in i+1:half_basis_size-1
            lblocks[indmap_2L[2,i,1,j]] = Elt(V[j,k,i,k]+V[k,j,k,i])
            lblocks[indmap_2L[2,i,1,j]] += Elt(V[k,j,i,k]+V[j,k,k,i])*(-2)
            lblocks[indmap_2L[1,i,2,j]] = Elt(V[i,k,j,k]+V[k,i,k,j])
            lblocks[indmap_2L[1,i,2,j]] += Elt(V[i,k,k,j]+V[k,i,j,k])*(-2)
        end
        push!(scal_blocks[half_basis_size],block2masks(lblocks,Elt(1),rblocks));        
        
        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); rblocks[indmap_2R[2,k,1,k]] = Elt(1);
        for i in 1:half_basis_size-1
            lblocks[indmap_2L[2,i,1,i]] = Elt(V[i,k,k,i]+V[k,i,i,k])*2
        end
        for i in 1:half_basis_size-1, j in i+1:half_basis_size-1
            lblocks[indmap_2L[2,i,1,j]] = Elt(V[j,k,i,k]+V[k,j,k,i])*(-2)
            lblocks[indmap_2L[2,i,1,j]] += Elt(V[k,j,i,k]+V[j,k,k,i])*2
            lblocks[indmap_2L[1,i,2,j]] = Elt(V[i,k,k,j]+V[k,i,j,k])*2
        end
        push!(op_blocks[half_basis_size],block2masks(lblocks,jkil_2,rblocks));

        
        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); rblocks[indmap_2R[1,k,1,k]] = Elt(1);
        for i in 1:half_basis_size-1
            lblocks[indmap_2L[2,i,2,i]] = Elt(V[k,k,i,i])
        end
        for i in 1:half_basis_size-1, j in i+1:half_basis_size-1
            lblocks[indmap_2L[2,i,2,j]] = Elt(V[k,k,j,i]+V[k,k,i,j])
        end
        push!(scal_blocks[half_basis_size],block2masks(lblocks,Elt(1),rblocks));        

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); rblocks[indmap_2R[2,k,2,k]] = Elt(1);
        for i in 1:half_basis_size-1
            lblocks[indmap_2L[1,i,1,i]] = Elt(V[i,i,k,k])
        end
        for i in 1:half_basis_size-1, j in i+1:half_basis_size-1
            lblocks[indmap_2L[1,i,1,j]] = Elt(V[j,i,k,k]+V[i,j,k,k])
        end
        push!(scal_blocks[half_basis_size],block2masks(lblocks,Elt(1),rblocks));        
    end
    
    for i in 1:half_basis_size-1
        
        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); lblocks[indmap_2L[2,i,2,i]] = Elt(1);
        for j in half_basis_size+1:basis_size,k in j+1:basis_size
            rblocks[indmap_2R[1,j,1,k]] = Elt(V[j,k,i,i]+V[k,j,i,i])
        end
        push!(scal_blocks[half_basis_size],block2masks(lblocks,Elt(1),rblocks));        
        

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); lblocks[indmap_2L[1,i,1,i]] = Elt(1);
        for j in half_basis_size+1:basis_size,k in j+1:basis_size
            rblocks[indmap_2R[2,j,2,k]] = Elt(V[i,i,j,k]+V[i,i,k,j])
        end
        push!(scal_blocks[half_basis_size],block2masks(lblocks,Elt(1),rblocks));        
        
        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); lblocks[indmap_2L[2,i,1,i]] = Elt(1);
        for j in half_basis_size+1:basis_size,k in j+1:basis_size
            rblocks[indmap_2R[1,j,2,k]] = Elt(V[j,i,i,k]+V[i,j,k,i])*(-2)
            rblocks[indmap_2R[1,j,2,k]] += Elt(V[i,j,i,k]+V[j,i,k,i])
            rblocks[indmap_2R[2,j,1,k]] = Elt(V[k,i,i,j]+V[i,k,j,i])*(-2)
            rblocks[indmap_2R[2,j,1,k]] += Elt(V[k,i,j,i]+V[i,k,i,j])

        end
        push!(scal_blocks[half_basis_size],block2masks(lblocks,Elt(1),rblocks));        

        lblocks = zeros(Elt,cnt+1); rblocks = zeros(Elt,cnt+1); lblocks[indmap_2L[2,i,1,i]] = Elt(1);
        for j in half_basis_size+1:basis_size,k in j+1:basis_size
            rblocks[indmap_2R[1,j,2,k]] = Elt(V[j,i,i,k]+V[i,j,k,i])*2
            rblocks[indmap_2R[2,j,1,k]] = Elt(V[k,i,i,j]+V[i,k,j,i])*2
            rblocks[indmap_2R[2,j,1,k]] += Elt(V[k,i,j,i]+V[i,k,i,j])*(-2)

        end
        push!(op_blocks[half_basis_size],block2masks(lblocks,jkil_2,rblocks));
    end
    
    #println("halfbasis $(sum(length.(op_blocks))+sum(length.(scal_blocks)))")

    op_blocks = map(op_blocks) do b
        filter!(b) do f
            sum(f[1])>0 && sum(f[end]) > 0
        end
    end

    scal_blocks = map(scal_blocks) do b
        filter!(b) do f
            sum(f[1])>0 && sum(f[end]) > 0
        end
    end

    opscal_blocks = Vector{FusedSparseBlock{Elt,O,typeof(psp)}}(undef,basis_size);

    for i in 1:basis_size
        vecs = Tuple{Vector{Bool},Vector{Elt},Union{Elt,O},Vector{Elt},Vector{Bool}}[];

        
        for (lm,lb,o,rb,rm) in op_blocks[i]
            @assert sum(lm) == length(lb)
            @assert sum(rm) == length(rb)
            (norm(lb) < 1e-12 || norm(rb) < 1e-12) && continue

            for sp in  domspaces[i,lm]
                @assert space(o,1) == sp
            end
            for sp in domspaces[mod1(i+1,end),rm]
                @assert space(o,4)' == sp
            end
            #@show o
            push!(vecs,(lm,lb,convert(Union{Elt,O},o),rb,rm));
        end
        for (lm,lb,o,rb,rm) in scal_blocks[i]
            @assert sum(lm) == length(lb)
            @assert sum(rm) == length(rb)

            (norm(lb) < 1e-12 || norm(rb) < 1e-12) && continue
            
            left_v = first(domspaces[i,lm]);
            for sp in domspaces[i,lm]
                @assert left_v == sp
            end
            for sp in domspaces[mod1(i+1,end),rm]
                @assert left_v == sp
            end

            #right_v = adjoint(first(domspaces[mod1(i+1,end),rm]))
            virt = isomorphism(Matrix{Elt},left_v,left_v);
            phys = isomorphism(Matrix{Elt},psp,psp);
            @plansor to[-1 -2;-3 -4] := virt[-1;1]*phys[-2;2]*τ[1 2;-3 -4]

            push!(vecs,(lm,lb,convert(Union{Elt,O},o*to),rb,rm));
        end
        opscal_blocks[i] = FusedSparseBlock{Elt,O,typeof(psp)}(domspaces[i,:],adjoint.(domspaces[mod1(i+1,end),:]),psp,vecs);
    end
    
    (compressed_ham,mapped) = compress(FusedMPOHamiltonian{Elt,O,typeof(psp)}(opscal_blocks));
    
    indmap_1Ls = copy.([indmap_1L for i in 1:length(compressed_ham)+1]);
    indmap_2Ls = copy.([indmap_2L for i in 1:length(compressed_ham)+1]);
    indmap_1Rs = copy.([indmap_1R for i in 1:length(compressed_ham)+1]);
    indmap_2Rs = copy.([indmap_2R for i in 1:length(compressed_ham)+1]);
    for (loc,m) in enumerate(mapped),
        symb in [indmap_1Ls,indmap_2Ls,indmap_1Rs,indmap_2Rs]
        
        for (i,el) in enumerate(symb[loc])
            hit = findfirst(x->x==el,m);
            if isnothing(hit)
                symb[loc][i] = 0
            else
                symb[loc][i] = hit;
            end
        end
    end


    compressed_ham,indmap_1Ls, indmap_1Rs, indmap_2Ls, indmap_2Rs
end