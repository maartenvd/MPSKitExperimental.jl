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

function MPSKit.∂∂AC2(pos::Int,mps,ham::FusedMPOHamiltonian{E,O,Sp},cache) where {E,O,Sp}
    opp1 = ham[pos];
    opp2 = ham[pos+1];
    le = leftenv(cache,pos,mps);
    re = rightenv(cache,pos+1,mps);
    
    # tranpose factories:
    S = spacetype(eltype(mps));
    storage = storagetype(eltype(mps));

    tvaltype_2_3 = TransposeFactType(S,storage,2,3);
    tvaltype_3_2 = TransposeFactType(S,storage,3,2);
    tvaltype_2_1 = TransposeFactType(S,storage,2,1);
    tvaltype_1_2 = TransposeFactType(S,storage,1,2);
    mvaltype_2_3 = DelayedFactType(S,storage,2,3);
    mvaltype_3_2 = DelayedFactType(S,storage,3,2);
    
    mfactory_2_3 = Dict{Any,mvaltype_2_3}();
    mfactory_3_2 = Dict{Any,mvaltype_3_2}();
    tfactory_2_3 = Dict{Any,tvaltype_2_3}();
    tfactory_3_2 = Dict{Any,tvaltype_3_2}();
    tfactory_2_1 = Dict{Any,tvaltype_2_1}();
    tfactory_1_2 = Dict{Any,tvaltype_1_2}();

    #---- create factories
    promise_creation = Dict{Any,Any}();

    let tfactory_2_1 = tfactory_2_1,
        mfactory_2_3 = mfactory_2_3,
        tfactory_3_2 = tfactory_3_2,
        mfactory_3_2 = mfactory_3_2
        
        for (lmask,lblock,e,rblock,rmask) in opp1.blocks
            l = le[lmask][1];

            homsp_2_1_2_1 = codomain(l)←domain(l);
            key_2_1_2_1 = (homsp_2_1_2_1,(3,1),(2,));

            if !(key_2_1_2_1 in keys(promise_creation))
                promise_creation[key_2_1_2_1] = @Threads.spawn (tfactory_2_1,TransposeFact(homsp_2_1_2_1,storage,(3,1),(2,)))
            end

            key_2_3_2_3 = space(l,3)*space(l,1)←space(e,3)'*space(e,4)'*space(e,2)';
            key_2_3_3_2 = (key_2_3_2_3,(2,5,4),(1,3));
            if !(key_2_3_3_2 in keys(promise_creation))
                t = @Threads.spawn (mfactory_2_3,DelayedFact(key_2_3_2_3,storage))
                promise_creation[key_2_3_2_3] = t
                promise_creation[key_2_3_3_2] = @Threads.spawn (tfactory_3_2,TransposeFact(fetch(t)[2],(2,5,4),(1,3)))
            end

            p1 = opp1.pspace;
            p2 = opp2.pspace;
            v1 = left_virtualspace(mps,pos-1);
            v2 = right_virtualspace(mps,pos+1);

            homsp_mult = v1*p1*space(e,4) ← v2*p2'
            if !(homsp_mult in keys(promise_creation))
                promise_creation[homsp_mult] = @Threads.spawn (mfactory_3_2,DelayedFact(homsp_mult,storage))
            end
        end

        for (lmask,lblock,e,rblock,rmask) in opp2.blocks
            r = re[rmask][1]
            homsp_2_1_1_2 = codomain(r)←domain(r);
            key_2_1_1_2 = (homsp_2_1_1_2,(2,),(1,3));
            
            if !(key_2_1_1_2 in keys(promise_creation))
                promise_creation[key_2_1_1_2] = @Threads.spawn (tfactory_1_2,TransposeFact(homsp_2_1_1_2,storage,(2,),(1,3)))
            end

            homsp_3_1_1_2 = space(e,3)*space(e,1)*space(e,2)←space(r,1)'*space(r,3)';
            key_3_2 = (homsp_3_1_1_2,(4,1,2),(5,3));
            if !(key_3_2 in keys(promise_creation))
                t = @Threads.spawn (mfactory_3_2,DelayedFact(homsp_3_1_1_2,storage))
                promise_creation[homsp_3_1_1_2] = t
                promise_creation[key_3_2] = @Threads.spawn (tfactory_3_2,TransposeFact(fetch(t)[2],(4,1,2),(5,3)))
            end

            p1 = opp1.pspace;
            p2 = opp2.pspace;
            v1 = left_virtualspace(mps,pos-1);
            v2 = right_virtualspace(mps,pos+1);

            homsp_3_2_2_3 = v1*p1*space(e,1)'←v2*p2';
            key_3_2_2_3 = (homsp_3_2_2_3,(1,2),(4,5,3));
            if !(key_3_2_2_3 in keys(promise_creation))
                promise_creation[key_3_2_2_3] = @Threads.spawn (tfactory_2_3,TransposeFact(homsp_3_2_2_3,storage,(1,2),(4,5,3)))
            end
        end


        for (k,t) in promise_creation
            (d,v) = fetch(t)
            Base.setindex!(d,v,k);
        end
    end

    process_left_blocks = Map() do (lmask,lblock,e,rblock,rmask)
        cl = le[lmask];
        
        l = rmul!(fast_copy(cl[1]),lblock[1])
        for i in 2:length(cl)
            l = fast_axpy!(lblock[i],cl[i],l);
        end

        homsp_2_1 = codomain(l)←domain(l);
        key_2_1 = (homsp_2_1,(3,1),(2,));
        
        l_transposed = tfactory_2_1[key_2_1](l);
        t_e = transpose(e,(1,),(3,4,2)); # should be cheap

        key_2_3 = space(l,3)*space(l,1)←space(e,3)'*space(e,4)'*space(e,2)';
        tl = mfactory_2_3[key_2_3]();
        mul!(tl,l_transposed,t_e);
        
        free!(tfactory_2_1[key_2_1],l_transposed)

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

        rt = tfactory_1_2[(codomain(r)←domain(r),(2,),(1,3))](r)
        et = transpose(e,(3,1,2),(4,));

        blocked = mfactory_3_2[codomain(et)←domain(rt)]();
        mul!(blocked,et,rt);
        
        free!(tfactory_1_2[(codomain(r)←domain(r),(2,),(1,3))],rt)

        (lmask,lblock,blocked)
    end

    blocked_right_blocks = tcollect(process_right_blocks,opp2.blocks)

    filter!(blocked_right_blocks) do (lmask,lblock,r)
        !isempty(lblock) && norm(r)>1e-12
    end

    left_group = Dict();
    right_group = Dict();

    for (i,(l,rblock,rmask)) in enumerate(blocked_left_blocks)
        left_group[space(l,4)'] = [i;get(left_group,space(l,4)',[])]
    end
    
    for (i,(lmask,lblock,r)) in enumerate(blocked_right_blocks)
        right_group[space(r,2)] = [i;get(right_group,space(r,2),[])]
    end

    mapper = Map() do (k)
        lefts = left_group[k];
        rights = right_group[k];
        
        cur_left_blocks = blocked_left_blocks[lefts];
        cur_right_blocks = blocked_right_blocks[rights];

        
        d = fill(zero(E),length(lefts),length(rights));
        for (i,(l,rblock,rmask)) in enumerate(cur_left_blocks),
            (j,(lmask,lblock,r)) in enumerate(cur_right_blocks)
            
            alpha = sum(rblock[lmask[rmask]].*lblock[rmask[lmask]]);
            d[i,j] = alpha
            
        end
        

        (U_s,R_s) = qr(d);
        U = Matrix(U_s);
        R = Matrix(R_s);

        flt = map(x-> norm(R[x,:])>1e-12,1:size(R,1))
        U = U[:,flt]
        R = R[flt,:]
        
        if size(R,1) == 0
            return []
        end
        
        
        (L_s,V_s) = lq(R)
        L = Matrix(L_s);
        V = Matrix(V_s)

        flt = map(x-> norm(L[:,x])>1e-12,1:size(L,2))
        V = V[flt,:]
        L = L[:,flt]
        
        if size(L,2) == 0
            return []
        end
        
        U = U*L;

        tot = Map() do i
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
            
            l_blocked = tfactory_3_2[(codomain(l)←domain(l),(2,5,4),(1,3))](l);
            r_blocked = tfactory_3_2[(codomain(r)←domain(r),(4,1,2),(5,3))](r)
            

            p1 = opp1.pspace;
            p2 = opp2.pspace;
            v1 = left_virtualspace(mps,pos-1);
            v2 = right_virtualspace(mps,pos+1);
            
            connector = space(r,2);

            homsp_mult = v1*p1*connector' ← v2*p2'
            temp_left = mfactory_3_2[homsp_mult];
            homsp_3_2_2_3 = v1*p1*connector'←v2*p2';
            key_3_2_2_3 = (homsp_3_2_2_3,(1,2),(4,5,3));
            temp_right = tfactory_2_3[key_3_2_2_3];

            # transpose + temps
            (l_blocked,temp_left,r_blocked,temp_right)
        end
    
        tcollect(tot,1:size(U,2))
    end
    
    blocks = reduce(vcat,tcollect(mapper,intersect(keys(right_group),keys(left_group))))
    fused_∂∂AC2(convert(Vector{typeof(blocks[1])},blocks))
end

#@tightloop_planar inner_ac2 y[-1 -2;-3 -4] := lblock[-1 -2;1 2 3]*x[1 2;4 5]*[4 5 3;-3 -4]
function _inner_ac2!(toret,x,l,temp,r,temp_trans)
    temp_1 = temp();
    mul!(temp_1,l,x);
    temp_2 = temp_trans(temp_1);
    free!(temp,temp_1)
    mul!(toret,temp_2,r,true,true)
    free!(temp_trans,temp_2)
end

function _reduce_ac2(blocks,x,basesize)
    if length(blocks) <= basesize
        toret = zero(x)
        for (l,temp,r,temp_trans) in blocks
            _inner_ac2!(toret,x,l,temp,r,temp_trans)
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
