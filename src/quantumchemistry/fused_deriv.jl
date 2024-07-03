struct fused_∂∂AC{A}
    blocks::A
end

function _make_AC_factories(opp::FusedSparseBlock{E,O,Sp},ac) where {E,O,Sp}
    S = spacetype(O);
    storage = storagetype(O);


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



    #---- create factories
    promise_creation = Dict{Any,Any}();

    let 
        
        for (lmask,lblock,e,rblock,rmask) in opp.blocks
            l_virt = space(ac,1);
            l_o = opp.domspaces[lmask][1];
            r_o = opp.imspaces[rmask][1];
            r_virt = space(ac,3)';

            homsp_2_1_2_1 = l_virt*l_o'←l_virt
            key_2_1_2_1 = (homsp_2_1_2_1,(3,1),(2,));

            if !(key_2_1_2_1 in keys(promise_creation))
                promise_creation[key_2_1_2_1] = @Threads.spawn (tfactory_2_1,TransposeFact(homsp_2_1_2_1,storage,(3,1),(2,)))
            end

            key_2_3_2_3 = l_virt'*l_virt←space(e,3)'*space(e,4)'*space(e,2)';
            key_2_3_3_2 = (key_2_3_2_3,(2,5,4),(1,3));
            if !(key_2_3_3_2 in keys(promise_creation))
                t_t = @Threads.spawn (mfactory_2_3,DelayedFact(key_2_3_2_3,storage))
                promise_creation[key_2_3_2_3] = t_t
                promise_creation[key_2_3_3_2] = @Threads.spawn (tfactory_3_2,TransposeFact(fetch(t_t)[2],(2,5,4),(1,3)))
            end

            p1 = opp.pspace;
            v1 = l_virt;
            v2 = r_virt;

            homsp_mult = v1*p1*space(e,4) ← v2
            if !(homsp_mult in keys(promise_creation))
                t_tt = @Threads.spawn (mfactory_3_1,DelayedFact(homsp_mult,storage))
                promise_creation[homsp_mult] = t_tt
                promise_creation[(homsp_mult,(1,2),(4,3))] = @Threads.spawn (tfactory_2_2,TransposeFact(fetch(t_tt)[2],(1,2),(4,3)))
            end

        end

        for (k,t) in promise_creation
            (d,v) = fetch(t)
            Base.setindex!(d,v,k);
        end
    end

     
    return (mfactory_2_3, mfactory_3_1, tfactory_2_1, tfactory_3_2, tfactory_2_2)

end

function MPSKit.∂∂AC(pos::Int,mps,ham::FusedMPOHamiltonian,cache)
    opp = ham[pos];
    le = leftenv(cache,pos,mps);
    re = rightenv(cache,pos,mps);



    # tranpose factories:
    (mfactory_2_3, mfactory_3_1, tfactory_2_1, tfactory_3_2, tfactory_2_2) = _make_AC_factories(opp,mps.AC[pos]);

    process_blocks = Map() do (lmask,lblock,e,rblock,rmask)
        cl = le[lmask];
        cr = re[rmask];

        
        l = rmul!(fast_copy(cl[1]),lblock[1])
        for i in 2:length(cl)
            l = fast_axpy!(lblock[i],cl[i],l);
        end

        r = rmul!(fast_copy(cr[1]),rblock[1])
        for i in 2:length(rblock)
            r = fast_axpy!(rblock[i],cr[i],r);
        end        

        e_t = transpose(e,(1,),(3,4,2));

        l_tf = tfactory_2_1[(codomain(l)←domain(l),(3,1),(2,))];
        l_t = l_tf(l);

        le_tf = mfactory_2_3[codomain(l_t)←domain(e_t)]

        le_t = le_tf();
        mul!(le_t,l_t,e_t);
        free!(l_tf,l_t);

        tle_f = tfactory_3_2[(codomain(le_t)←domain(le_t),(2,5,4),(1,3))];
        tle = tle_f(le_t);
        free!(le_tf,le_t);

        temp_mul = mfactory_3_1[codomain(tle)←space(r,1)];
        temp_trans = tfactory_2_2[(codomain(tle)←space(r,1),(1,2),(4,3))]


        (tle,temp_mul,temp_trans,r)
    end


    blocks = tcollect(process_blocks,opp.blocks)

    filter!(blocks) do (tle,temp_mul,temp_trans,r)
        !(norm(tle)<1e-12 || norm(r)<1e-12)
    end
    
    fused_∂∂AC(blocks)
end

function (h::fused_∂∂AC)(x)
    @floop for (tle,temp_mul,temp_trans,r) in h.blocks
        @init t = similar(x)

        t1 = temp_mul();
        mul!(t1,tle,x)
        t2 = temp_trans(t1);
        free!(temp_mul,t1);
        mul!(t,t2,r);
        free!(temp_trans,t2);

        @reduce() do (toret = zero(x); t)
            fast_axpy!(true,t,toret);
            toret
        end
    end
    
    toret
end

Base.:*(a::fused_∂∂AC,v) = a(v)

# ugly - inconsistent with MPOHamiltonian
MPSKit.expectation_value(st::FiniteMPS,th::FusedMPOHamiltonian,envs = environments(st,th)) =
    dot(st.AC[1],MPSKit.∂∂AC(1,st,th,envs)(st.AC[1]))/dot(st.AC[1],st.AC[1])


struct fused_∂∂AC2{A}
    blocks::A
end

@tightloop_planar leftblock_ac2 allocator=malloc out[-1 -2 -3;-4 -5] := l[-1 1;-4]*o[1 -2;-5 -3]
@tightloop_planar rightblock_ac2 allocator=malloc out[-1 -2 -3;-4 -5] := r[-1 1;-4]*o[-3 -5;-2 1]
@tightloop_planar ac2_update allocator=malloc out[-1 -2;-3 -4] += l[-1 -2 5;1 2]*ac2[1 2;3 4]*r[3 4 5;-3 -4]

function MPSKit.∂∂AC2(pos::Int,mps,ham::FusedMPOHamiltonian{E,O,Sp},cache) where {E,O,Sp}
    opp1 = ham[pos];
    opp2 = ham[pos+1];
    le = leftenv(cache,pos,mps);
    re = rightenv(cache,pos+1,mps);
    begin
        
        # tranpose factories:
        S = spacetype(eltype(mps));
        storage = storagetype(eltype(mps));

        left_example = le[1]
        right_example = re[1]
        o_example_space = space(left_example,2)*opp1.pspace←opp1.pspace*space(left_example,2)
        o_example_type = tensormaptype(S,2,2,storage)
        leftblock_example = leftblock_ac2(l=(typeof(left_example),space(left_example)),o = (o_example_type,o_example_space))
        rightblock_example = rightblock_ac2(r=(typeof(right_example),space(right_example)),o = (o_example_type,o_example_space))
        leftblock_factories = Dict{typeof(o_example_space),typeof(leftblock_example)}()
        rightblock_factories = Dict{typeof(o_example_space),typeof(rightblock_example)}()
        promise_creation = Dict{typeof(o_example_space),Any}()
        for (lmask,lblock,e,rblock,rmask) in opp1.blocks
            space(e) in keys(promise_creation) && continue
            promise_creation[space(e)] = @Threads.spawn begin
                v_example = le[findfirst(lmask)]
                leftblock_ac2(l=(typeof(v_example),space(v_example)),o = (typeof(e),space(e)))
            end
        end
        for (k,v) in promise_creation
            leftblock_factories[k] = fetch(v)
        end

        empty!(promise_creation)

        promise_creation = Dict{typeof(o_example_space),Any}()
        for (lmask,lblock,e,rblock,rmask) in opp2.blocks
            space(e) in keys(promise_creation) && continue
            promise_creation[space(e)] = @Threads.spawn begin
                v_example = re[findfirst(rmask)]
                rightblock_ac2(r=(typeof(v_example),space(v_example)),o = (typeof(e),space(e)))
            end
        end
        for (k,v) in promise_creation
            rightblock_factories[k] = fetch(v)
        end
    end
    process_left_blocks = Map() do (lmask,lblock,e,rblock,rmask)
        cl = le[lmask];
        
        l = rmul!(fast_copy(cl[1]),lblock[1])
        for i in 2:length(cl)
            l = fast_axpy!(lblock[i],cl[i],l);
        end

        tl = leftblock_factories[space(e)](l=l,o=e)
        (tl,rblock,rmask)
    end
    
    blocked_left_blocks = tcollect(process_left_blocks,opp1.blocks)

    filter!(blocked_left_blocks) do (l,rblock,rmask)
        norm(l)>1e-12 && !isempty(rblock)
    end

    process_right_blocks = Map() do (lmask,lblock,e,rblock,rmask)
        cr = re[rmask];

        r = rmul!(fast_copy(cr[1]),rblock[1])
        for i in 2:length(rblock)
            r = fast_axpy!(rblock[i],cr[i],r)
        end

        rl = rightblock_factories[space(e)](r=r,o=e)
        
        (lmask,lblock,rl)
    end

    blocked_right_blocks = tcollect(process_right_blocks,opp2.blocks)

    filter!(blocked_right_blocks) do (lmask,lblock,r)
        !isempty(lblock) && norm(r)>1e-12
    end

    left_group = Dict{typeof(opp1.pspace),Vector{Int}}();
    right_group = Dict{typeof(opp1.pspace),Vector{Int}}();

    for (i,(l,rblock,rmask)) in enumerate(blocked_left_blocks)
        left_group[space(l,3)'] = [i;get(left_group,space(l,3)',Int[])]
    end
    
    for (i,(lmask,lblock,r)) in enumerate(blocked_right_blocks)
        right_group[space(r,3)] = [i;get(right_group,space(r,3),Int[])]
    end

    p1 = opp1.pspace;
    p2 = opp2.pspace;
    v1 = left_virtualspace(mps,pos-1);
    v2 = right_virtualspace(mps,pos+1);
    ac2_type = tensormaptype(S,2,2,storage)
    ac2_structure = v1*p1 ← v2*(p2)'

    common_keys = collect(intersect(keys(left_group),keys(right_group)))
    d_matrices = Dict(map(k-> k=> fill(zero(E),length(left_group[k]),length(right_group[k])),common_keys))
    splats = reduce(vcat,[
        reduce(vcat,[
            [(k,(il,lv),(ir,rv)) for (ir,rv) in enumerate(get(right_group,k,Int[]))] 
            for (il,lv) in enumerate(v)]) 
                for (k,v) in left_group])

    @threads for (k,(il,lv),(ir,rv)) in splats
        (l,rblock,rmask) = blocked_left_blocks[lv]
        (lmask,lblock,r) = blocked_right_blocks[rv]
        d_matrices[k][il,ir] = sum(rblock[lmask[rmask]].*lblock[rmask[lmask]]);
    end

    # this bit is single threaded but should take almost no time
    pairs = map(collect(keys(d_matrices))) do k
        d = d_matrices[k]
        
        (U_s,R_s) = qr(d);
        U = Matrix(U_s);
        R = Matrix(R_s);

        flt = map(x-> norm(R[x,:])>1e-12,1:size(R,1))
        U = U[:,flt]
        R = R[flt,:]

        if size(R,1)==0 || size(R,2) == 0
            return k=> (U,R)
        end

        (L_s,V_s) = lq(R)
        L = Matrix(L_s);
        V = Matrix(V_s)

        flt = map(x-> norm(L[:,x])>1e-12,1:size(L,2))
        V = V[flt,:]
        L = L[:,flt]
        
        U = U*L;

        k => (U,V)
    end
    filter!(pairs) do x
        (k,(a,b)) = x
        size(a,2) > 0
    end
    kvs = Dict(pairs)


    totblock_inds = reduce(vcat,[[(k,i) for i in 1:size(kvs[k][1],2)] for k in keys(kvs)])

    
    mapper = Map() do (k,i)
        (U,V) = kvs[k]
        cur_left_blocks = blocked_left_blocks[left_group[k]]
        cur_right_blocks = blocked_right_blocks[right_group[k]]

        l = rmul!(fast_copy(cur_left_blocks[1][1]),U[1,i]);
        for j in 2:size(U,1)
            u = U[j,i];
            if abs(u)>1e-12
                fast_axpy!(u,cur_left_blocks[j][1],l)
            end
        end
        
        r = rmul!(fast_copy(cur_right_blocks[1][3]),V[i,1]);
        for j in 2:size(V,2)
            v = V[i,j];
            if abs(v)>1e-12
                fast_axpy!(v,cur_right_blocks[j][3],r)
            end
        end
        


        factory = ac2_update(l = (typeof(l),space(l)), r = (typeof(r),space(r)), ac2 = (ac2_type,ac2_structure), out = (ac2_type,ac2_structure))
        #fast_tmp_1 = fast_init(codomain(l),codomain(r),storagetype(l))
        #fast_submult = LeftSubMult(space(l),ac2_structure)
        
        # transpose + temps
        #(l,r,(fast_tmp_1,fast_submult))
        (l,r,factory)
    end

    blocks = tcollect(mapper,totblock_inds)
    
    fused_∂∂AC2(convert(Vector{typeof(blocks[1])},blocks))
    
end

function _reduce_ac2(blocks,x,basesize)
    if length(blocks) <= basesize
        toret = zero(x)

       
               #=

        for (l,r,(fast_tmp_1,fast_submult)) in blocks
            tmp = fast_tmp_1(MallocBackend(),true)
            rmul!(tmp,false)
            fast_submult(tmp,l,x)
            mul!(toret,tmp,r,true,true)
            TensorOperations.tensorfree!(tmp, MallocBackend())
        end
  =#
        for (l,r,factory) in blocks
            factory(l=l,r=r,ac2=x,out=toret)
        end


        return toret
    else

        spl = Int(ceil(length(blocks)/2));
        t = @Threads.spawn _reduce_ac2(blocks[1:spl],x,basesize)
        toret = _reduce_ac2(view(blocks,spl+1:length(blocks)),x,basesize)
        fast_axpy!(true,fetch(t),toret)
        return toret
    end
end

function (h::fused_∂∂AC2)(x)
    _reduce_ac2(h.blocks,x,ceil(length(h.blocks)/nthreads()))
end
