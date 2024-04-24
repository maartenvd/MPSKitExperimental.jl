#=
    we can surpisingly enough hook into the standard finite env!
=#
function MPSKit.environments(state::FiniteMPS{S},ham::FusedMPOHamiltonian) where S
    lll = l_LL(state);rrr = r_RR(state)
    rightstart = Vector{S}();leftstart = Vector{S}()

    for (i,sp) in enumerate(ham[1].domspaces)
        util_left = Tensor(x->storagetype(S)(undef,x),sp'); fill_data!(util_left,one);
        @plansor ctl[-1 -2; -3]:= lll[-1;-3]*util_left[-2]
        
        if i != 1
            ctl = zero(ctl)
        end

        push!(leftstart,ctl)
    end

    for (i,sp) in enumerate(ham[length(state)].imspaces)
        util_right = Tensor(x->storagetype(S)(undef,x),sp'); fill_data!(util_right,one);
        @plansor ctr[-1 -2; -3]:= rrr[-1;-3]*util_right[-2]

        if i != length(ham[length(state)].imspaces)
            ctr = zero(ctr)
        end

        push!(rightstart,ctr)
    end

    return disk_environments(state,ham,leftstart,rightstart)
end
#=
    need to define the relevant transfer operators
=#

function _make_ltrans_factories(opp,A)
    S = spacetype(A);
    storage = storagetype(A);
    
    tvaltype_1_2 = TransposeFactType(S,storage,1,2);
    tvaltype_2_2 = TransposeFactType(S,storage,2,2);
    mvaltype_2_2 = DelayedFactType(S,storage,2,2);
    mvaltype_2_1 = DelayedFactType(S,storage,2,1);
    
    tfactory_2_2 = Dict{Any,tvaltype_2_2}(); 
    mfactory_2_2 = Dict{Any,mvaltype_2_2}();
    mfactory_2_1 = Dict{Any,mvaltype_2_1}();
    tfactory_1_2 = Dict{Any,tvaltype_1_2}();
    
    v_1 = space(A,1);
    v_3 = space(A,3)';
    p = space(A,2);

    promise_creation = Dict{Any,Any}();

    for (lmask,lblock,e,rblock,rmask) in opp.blocks
        #r_hit = first(v[lmask]);
        if e isa TensorMap
            mpo_virt_1 = space(e,1);
            mpo_virt_4 = space(e,4)';
        else
            mpo_virt_1 = first(opp.domspaces[lmask]) #space(r_hit,2)';
            mpo_virt_4 = first(opp.domspaces[lmask])
        end
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

    end
    
    for (k_t,t_t) in promise_creation
        (d_t,v_t) = fetch(t_t)
        Base.setindex!(d_t,v_t,k_t);
    end

    return (tfactory_2_2, mfactory_2_2, mfactory_2_1,tfactory_1_2);
end

@tightloop_planar left_trans y[-1 -2;-3] := v[4 2;1]*A[1 3;-3]*O[2 5;3 -2]*Ab[-1 5;4]
mpotype(O::FusedSparseBlock{E,H,Sp}) where {E,H,Sp} = H

function MPSKit.transfer_left(v::Vector,O::FusedSparseBlock,A,Ab=A)
    Ab_flipped = convert(TensorMap,transpose(Ab',(1,3),(2,)));
    
    homspace_example = O.pspace*O.pspace←O.pspace*O.pspace
    homspace_type = typeof(homspace_example)
    factory_type = typeof(left_trans(v=(typeof(v[1]),space(v[1])),A = (typeof(A),space(A)),Ab = (typeof(Ab_flipped),space(Ab_flipped)), O = (mpotype(O),homspace_example)))
    factories = Dict{homspace_type,factory_type}();
    promise_creation = Dict{homspace_type,Any}();

    for (lmask,lblock,e,rblock,rmask) in O.blocks
        e isa Number && @assert false #not supported anymore
        
        space(e) in keys(promise_creation) && continue #someone else will make this element
        promise_creation[space(e)] = @Threads.spawn begin
            v_example = v[findfirst(lmask)]
            left_trans(v = (typeof(v_example),space(v_example)),A = (typeof(A),space(A)),Ab = (typeof(Ab_flipped),space(Ab_flipped)), O = (typeof(e),space(e)))
   
        end
    end

    for (k,v) in promise_creation
        factories[k] = fetch(v)
    end

    mapper = Map() do (lmask,lblock,e,rblock,rmask)
        # reduce left
        v_masked = v[lmask];

        l = rmul!(fast_copy(v_masked[1]),lblock[1]);
        for i in 2:length(v_masked)
            l = fast_axpy!(lblock[i],v_masked[i],l);
        end

        nl = factories[space(e)](v = l, A = A, Ab = Ab_flipped,O = e)
        
        # expand r
        toret = map(rblock) do r
            (r,nl)
        end
        (toret,rmask)
    end

    mapped = tcollect(mapper,O.blocks)
    out = Vector{eltype(v)}(undef,length(O.imspaces))
    isassigned = fill(false,length(O.imspaces));

    for i in 1:length(O.imspaces)
        for (lb,lm) in mapped
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

function _make_rtrans_factories(opp,A)
    S = spacetype(A);
    storage = storagetype(A);

    tvaltype_1_2 = TransposeFactType(S,storage,1,2);
    tvaltype_2_2 = TransposeFactType(S,storage,2,2);
    mvaltype_2_2 = DelayedFactType(S,storage,2,2);
    mvaltype_2_1 = DelayedFactType(S,storage,2,1);

    tfactory_1_2 = Dict{Any,tvaltype_1_2}(); # transpose r
    tfactory_2_2 = Dict{Any,tvaltype_2_2}();
    mfactory_2_2 = Dict{Any,mvaltype_2_2}();
    mfactory_2_1 = Dict{Any,mvaltype_2_1}();

    v_1 = space(A,1);
    v_3 = space(A,3)';
    p = space(A,2);

    promise_creation = Dict{Any,Any}()
    

    for (lmask,lblock,e,rblock,rmask) in opp.blocks

        #r_hit = first(v[rmask]);
        if e isa TensorMap
            mpo_virt_1 = space(e,1);
            mpo_virt_4 = space(e,4)';
        else
            mpo_virt_1 = first(opp.imspaces[rmask])'#space(r_hit,2);
            mpo_virt_4 = first(opp.imspaces[rmask])';
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

    return (tfactory_2_2, mfactory_2_2, mfactory_2_1,tfactory_1_2);

end

function MPSKit.transfer_right(v::Vector,O::FusedSparseBlock,A,Ab=A)
    # first we should do a pass through O/v for the factories, which can then be utilized in parallel
    Ab_flipped = transpose(Ab',(1,3),(2,));

    
    (tfactory_2_2, mfactory_2_2, mfactory_2_1,tfactory_1_2) = _make_rtrans_factories(O,A)


    mapper = Map() do (lmask,lblock,e,rblock,rmask)
        v_masked = v[rmask];


        r = rmul!(fast_copy(v_masked[1]),rblock[1]);
        for i in 2:length(rblock)
            r = fast_axpy!(rblock[i],v_masked[i],r)
        end

        rt = tfactory_1_2[(codomain(r) ← domain(r),(1,),(3,2))](r)

        ar = mfactory_2_2[codomain(A)←domain(rt)]()
        mul!(ar,A,rt);
        
        free!(tfactory_1_2[(codomain(r) ← domain(r),(1,),(3,2))],rt)

        ar_t = tfactory_2_2[(codomain(ar)←domain(ar),(2,4),(1,3))](ar)
        if e isa AbstractTensorMap
            ear = mfactory_2_2[codomain(e)←domain(ar_t)]();
            mul!(ear,e,ar_t);
        else
            @assert false # not implemented
            @plansor ear[-1 -2;-3 -4] = τ[-1 -2;1 2]*ar_t[1 2;-3 -4]
            mul!(ear,e,ear);
        end

        free!(mfactory_2_2[codomain(A)←domain(rt)],ar);

        ear_t = tfactory_2_2[(codomain(ear)←domain(ear),(3,1),(4,2))](ear);
        free!(tfactory_2_2[(codomain(ar)←domain(ar),(2,4),(1,3))],ar_t)

        nr = mfactory_2_1[codomain(ear_t)←domain(Ab_flipped)]()
        mul!(nr,ear_t,Ab_flipped)
        free!(tfactory_2_2[(codomain(ear)←domain(ear),(3,1),(4,2))],ear_t)

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

    for i in findall(!,isassigned)
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
