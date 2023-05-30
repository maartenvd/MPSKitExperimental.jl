#=
    we can surpisingly enough hook into the standard finite env!
=#
function MPSKit.environments(state::FiniteMPS{S},ham::FusedMPOHamiltonian,workdir = tempdir()) where S
    @assert false
    lll = l_LL(state);rrr = r_RR(state)
    rightstart = Vector{S}();leftstart = Vector{S}()

    for i in 1:ham.odim
        util_left = Tensor(x->storagetype(S)(undef,x),ham[1].domspaces[i]'); fill_data!(util_left,one);
        util_right = Tensor(x->storagetype(S)(undef,x),ham[length(state)].imspaces[i]'); fill_data!(util_right,one);

        @plansor ctl[-1 -2; -3]:= lll[-1;-3]*util_left[-2]
        @plansor ctr[-1 -2; -3]:= rrr[-1;-3]*util_right[-2]

        if i != 1
            ctl = zero(ctl)
        end

        if i != ham.odim
            ctr = zero(ctr)
        end

        push!(leftstart,ctl)
        push!(rightstart,ctr)
    end

    return environments(state,ham,leftstart,rightstart)
end
#=
    need to define the relevant transfer operators
=#

function MPSKit.transfer_left(v::Vector,O::FusedSparseBlock,A,Ab=A)
    Ab_flipped = transpose(Ab',(1,3),(2,));
    
    S = spacetype(A);
    storage = storagetype(A);
    
    tvaltype_1_2 = TransposeFactType(S,storage,1,2);
    tvaltype_2_2 = TransposeFactType(S,storage,2,2);
    mvaltype_2_2 = DelayedFactType(S,storage,2,2);
    mvaltype_2_1 = DelayedFactType(S,storage,2,1);
    
    mfactory_1_2 = Dict{Any,mvaltype_2_2}(); # multiply A with r
    tfactory_2_2 = Dict{Any,tvaltype_2_2}(); 
    mfactory_2_2 = Dict{Any,mvaltype_2_2}();
    mfactory_2_1 = Dict{Any,mvaltype_2_1}();
    tfactory_1_2 = Dict{Any,tvaltype_1_2}();
    
    v_1 = space(A,1);
    v_3 = space(A,3)';
    p = space(A,2);
    fact_block = map(O.blocks) do (lmask,lblock,e,rblock,rmask)
        r_hit = first(v[lmask]);
        if e isa TensorMap
            mpo_virt_1 = space(e,1);
            mpo_virt_4 = space(e,4)';
        else
            mpo_virt_1 = space(r_hit,2)';
            mpo_virt_4 = space(r_hit,2)';
        end
        temp0_homsp = v_1*mpo_virt_1'←v_1;
        if !((temp0_homsp,(1,),(3,2)) in keys(tfactory_1_2))
            tfactory_1_2[(temp0_homsp,(1,),(3,2))] = TransposeFact(temp0_homsp,storage,(1,),(3,2));
        end
        temp_0 = tfactory_1_2[(temp0_homsp,(1,),(3,2))];

        temp1_homsp = v_3*p'←v_1*mpo_virt_1;
        if !(temp1_homsp in keys(mfactory_1_2))
            mfactory_1_2[temp1_homsp] = DelayedFact(temp1_homsp,storage)
        end
        temp_1 = mfactory_1_2[temp1_homsp]
        

        temp2_homsp = temp1_homsp;
        if !((temp2_homsp,(3,1),(4,2)) in keys(tfactory_2_2))
            tfactory_2_2[(temp2_homsp,(3,1),(4,2))] = TransposeFact(temp_1,(3,1),(4,2));
        end
        temp_2 = tfactory_2_2[(temp2_homsp,(3,1),(4,2))];

        temp3_homsp = v_1'*v_3←p*mpo_virt_4
        if !(temp3_homsp in keys(mfactory_2_2))
            mfactory_2_2[temp3_homsp] = DelayedFact(temp3_homsp,storage);
        end
        temp_3 = mfactory_2_2[temp3_homsp]

        temp4_homsp = temp3_homsp;
        if !((temp4_homsp,(2,4),(1,3)) in keys(tfactory_2_2))
            tfactory_2_2[(temp4_homsp,(2,4),(1,3))] = TransposeFact(temp_3,(2,4),(1,3));
        end
        temp_4 = tfactory_2_2[(temp4_homsp,(2,4),(1,3))];

        temp_5_homsp = v_3*mpo_virt_4'←v_3;
        if !(temp_5_homsp in keys(mfactory_2_1))
            mfactory_2_1[temp_5_homsp] = DelayedFact(temp_5_homsp,storage);
        end
        temp_5 = mfactory_2_1[temp_5_homsp]
        
        (lmask,lblock,e,rblock,rmask,(temp_0,temp_1,temp_2,temp_3,temp_4,temp_5))
    end
    
    mapper = Map() do (lmask,lblock,e,rblock,rmask,(temp_0,temp_1,temp_2,temp_3,temp_4,temp_5))
        # reduce left
        v_masked = v[lmask];

        l = rmul!(fast_copy(v_masked[1]),lblock[1]);
        for i in 2:length(v_masked)
            l = fast_axpy!(lblock[i],v_masked[i],l);
        end

        l_perm = temp_0(l);
        lAb = temp_1();
        mul!(lAb,Ab_flipped,l_perm);

        lAb_perm = temp_2(lAb);
        
        lAbe = temp_3();
        if e isa AbstractTensorMap
            mul!(lAbe,lAb_perm,e);
        else
            @plansor lAbe[-1 -2;-3 -4] = lAb_perm[-1 -2;1 2]*τ[1 2;-3 -4]
            mul!(lAbe,e,lAbe);
        end

        lAbe_perm = temp_4(lAbe);

        nl = temp_5();
        mul!(nl,lAbe_perm,A);

        # expand r
        toret = map(rblock) do r
            (r,nl)
        end

        (toret,rmask)
    end

    mapped = tcollect(mapper,fact_block)#,basesize=1)
    
    out = Vector{eltype(v)}(undef,length(O.imspaces))
    isassigned = fill(false,length(O.imspaces));

    for (rb,rm) in mapped
        for ((scal,a),b) in zip(rb,findall(rm))
            if isassigned[b]
                fast_axpy!(scal,a,out[b]);
            else
                out[b] = rmul!(fast_copy(a),scal)
            end
        end

        isassigned[rm] .=true;
    end

    for i in findall(!,isassigned)
        # fill in
        homsp = space(Ab,3)'*O.imspaces[i]←space(A,3)'
        if !(homsp in keys(mfactory_2_1))
            mfactory_2_1[homsp] = DelayedFact(homsp,storage);
        end
        out[i] = mfactory_2_1[homsp]();
        mul!(out[i],false,out[i]);
    end
    
    out
end

function MPSKit.transfer_right(v::Vector,O::FusedSparseBlock,A,Ab=A)
    # first we should do a pass through O/v for the factories, which can then be utilized in parallel
    Ab_flipped = transpose(Ab',(1,3),(2,));

    S = spacetype(A);
    storage = storagetype(A);

    tvaltype_1_2 = TransposeFactType(S,storage,1,2);
    tvaltype_2_2 = TransposeFactType(S,storage,2,2);
    mvaltype_2_2 = DelayedFactType(S,storage,2,2);
    mvaltype_2_1 = DelayedFactType(S,storage,2,1);

    tfactory_1_2 = Dict{Any,tvaltype_1_2}(); # transpose r
    mfactory_1_2 = Dict{Any,mvaltype_2_2}(); # multiply A with r
    tfactory_2_2 = Dict{Any,tvaltype_2_2}();
    mfactory_2_2 = Dict{Any,mvaltype_2_2}();
    mfactory_2_1 = Dict{Any,mvaltype_2_1}();

    v_1 = space(A,1);
    v_3 = space(A,3)';
    p = space(A,2);

    promise_creation = Dict{Any,Any}()
    

    
    for (lmask,lblock,e,rblock,rmask) in O.blocks
        r_hit = first(v[rmask]);
        if e isa TensorMap
            mpo_virt_1 = space(e,1);
            mpo_virt_4 = space(e,4)';
        else
            mpo_virt_1 = space(r_hit,2);
            mpo_virt_4 = space(r_hit,2);
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

        temp_6_homsp = v_1*mpo_virt_1←v_1;
        if !(temp_6_homsp in keys(promise_creation))
            promise_creation[temp_6_homsp] = @Threads.spawn (mfactory_2_1,DelayedFact(temp_6_homsp,storage))
        end
    end

    for (k_t,t_t) in promise_creation
        (d_t,v_t) = fetch(t_t)
        Base.setindex!(d_t,v_t,k_t);
    end


    mapper = Map() do (lmask,lblock,e,rblock,rmask)
        v_masked = v[rmask];


        r = rmul!(fast_copy(v_masked[1]),rblock[1]);
        for i in 2:length(rblock)
            r = fast_axpy!(rblock[i],v_masked[i],r)
        end

        rt = tfactory_1_2[(codomain(r) ← domain(r),(1,),(3,2))](r)

        ar = mfactory_2_2[codomain(A)←domain(rt)]()
        mul!(ar,A,rt);

        ar_t = tfactory_2_2[(codomain(ar)←domain(ar),(2,4),(1,3))](ar)
        if e isa AbstractTensorMap
            ear = mfactory_2_2[codomain(e)←domain(ar_t)]();
            mul!(ear,e,ar_t);
        else
            @assert false # not implemented
            @plansor ear[-1 -2;-3 -4] = τ[-1 -2;1 2]*ar_t[1 2;-3 -4]
            mul!(ear,e,ear);
        end

        ear_t = tfactory_2_2[(codomain(ear)←domain(ear),(3,1),(4,2))](ear);
        
        nr = mfactory_2_1[codomain(ear_t)←domain(Ab_flipped)]()
        mul!(nr,ear_t,Ab_flipped)

        # expand r
        toret = map(lblock) do t
            (t,nr)
        end

        (lmask,toret)
    end

    mapped = tcollect(mapper,O.blocks)#,basesize=1)

    out = Vector{eltype(v)}(undef,length(O.domspaces))
    isassigned = fill(false,length(O.domspaces));
    @floop for i in 1:length(O.domspaces)
        for (lm,lb) in mapped
            lm[i] || continue
            (ct,cnr) = lb[sum(lm[1:i])];

            if isassigned[i]
                out[i] = fast_axpy!(ct,cnr,out[i])
            else
                out[i] = rmul!(fast_copy(cnr),ct)
                isassigned[i] = true
            end
        end
    end

    @floop for i in findall(!,isassigned)
        # fill in
        homsp = space(A,1)*O.domspaces[i]←space(Ab,1);

        if !(homsp in keys(mfactory_2_1))
            mfactory_2_1[homsp] = DelayedFact(homsp,storage);
        end

        out[i] = mfactory_2_1[homsp]()
        mul!(out[i],false,out[i])

    end

    out
end



#=
function MPSKit.transfer_right(v::Vector,O::FusedSparseBlock,A,Ab=A)
    A_flipped = transpose(A,(1,),(3,2));
    Ab_flipped = A_flipped';

    promise_creation = Dict{Any,Any}();

    S = spacetype(A);
    storage = storagetype(A);
    
    tvaltype_1_2 = TransposeFactType(S,storage,1,2);
    mvaltype_3_2 = DelayedFactType(S,storage,3,2);
    mvaltype_2_1 = DelayedFactType(S,storage,2,1);
    mvaltype_1_3 = DelayedFactType(S,storage,1,3);
    tvaltype_2_3 = TransposeFactType(S,storage,2,3);
    tvaltype_2_2 = TransposeFactType(S,storage,2,2);

    tfactory_1_2 = Dict{Any,tvaltype_1_2}(); # transpose r
    mfactory_3_2 = Dict{Any,mvaltype_3_2}();
    mfactory_2_1 = Dict{Any,mvaltype_2_1}();
    tfactory_2_3 = Dict{Any,tvaltype_2_3}();
    mfactory_1_3 = Dict{Any,mvaltype_1_3}();
    tfactory_2_2 = Dict{Any,tvaltype_2_2}();


    v_1 = space(A,1);
    v_3 = space(A,3)';
    p = space(A,2);


    # spin up tasks for factories
    for (lmask,lblock,e,rblock,rmask) in O.blocks
        o_4 = O.imspaces[rmask][1]'
        o_1 = O.domspaces[lmask][1]

        homsp_2_1_1_2 = v_3*o_4←v_3;
        key_2_1_1_2 = (homsp_2_1_1_2,(2,),(1,3));
        
        if !(key_2_1_1_2 in keys(promise_creation))
            promise_creation[key_2_1_1_2] = @Threads.spawn (tfactory_1_2,TransposeFact(homsp_2_1_1_2,storage,(2,),(1,3)))
        end

        homsp_3_2 = space(e,3)*space(e,1)*space(e,2)←v_3'*v_3;
        key_3_2 = (homsp_3_2,(4,1),(5,3,2))
        if !(key_3_2 in keys(promise_creation))
            t = @Threads.spawn (mfactory_3_2,DelayedFact(homsp_3_2,storage))
            promise_creation[homsp_3_2] = t
            promise_creation[key_3_2] = @Threads.spawn (tfactory_2_3,TransposeFact(fetch(t)[2],(4,1),(5,3,2)))
        end

        homsp_2_1_t = v_1*o_1←v_1
        if !(homsp_2_1_t in keys(promise_creation))
            promise_creation[homsp_2_1_t] = @Threads.spawn (mfactory_2_1,DelayedFact(homsp_2_1_t,storage))
        end

        homsp_1_3 = v_1←v_3*space(e,2)'*space(e,1)';
        key_2_2 = (homsp_1_3,(1,4),(2,3))
        if !(key_2_2 in keys(promise_creation))
            t_1_3 = @Threads.spawn (mfactory_1_3,DelayedFact(homsp_1_3,storage))
            
            promise_creation[homsp_1_3] = t_1_3
            promise_creation[key_2_2] = @Threads.spawn (tfactory_2_2,TransposeFact(fetch(t_1_3)[2],(1,4),(2,3)));
        end
    end


    for (k_t,t_t) in promise_creation
        (d_t,v_t) = fetch(t_t)
        Base.setindex!(d_t,v_t,k_t);
    end

    process_right_blocks = Map() do (lmask,lblock,e,rblock,rmask)
        cr = v[rmask];

        r = rmul!(fast_copy(cr[1]),rblock[1])
        for i in 2:length(rblock)
            r = fast_axpy!(rblock[i],cr[i],r)
        end

        rt = tfactory_1_2[(codomain(r)←domain(r),(2,),(1,3))](r)
        et = transpose(e,(3,1,2),(4,));

        blocked = mfactory_3_2[codomain(et)←domain(rt)]();
        mul!(blocked,et,rt);

        blt = tfactory_2_3[(codomain(blocked)←domain(blocked),(4,1),(5,3,2))](blocked);

        (lmask,lblock,blt)
    end
    
    @timeit to "rb" right_blocks = tcollect(process_right_blocks,O.blocks)

    
    calcu = Map() do odim
        outvec = mfactory_2_1[v_1*O.domspaces[odim]←v_1]();
        rmul!(outvec,false)
        
        for (lmask,lblock,blocked) in right_blocks
            lmask[odim] || continue
            
            α = lblock[sum(lmask[1:odim])];

            y_t = mfactory_1_3[codomain(A_flipped)←domain(blocked)]();
            mul!(y_t,A_flipped,blocked);

            ty_t = tfactory_2_2[(codomain(y_t) ← domain(y_t),(1,4),(2,3))](y_t);

            mul!(outvec,ty_t,Ab_flipped,α,true)
        end
        
        outvec
    end

    tcollect(calcu,1:length(O.domspaces))
end

=#