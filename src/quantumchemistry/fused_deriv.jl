struct fused_∂∂AC{A}
    blocks::A
end

function MPSKit.∂∂AC(pos::Int,mps,ham::FusedMPOHamiltonian,cache)
    opp = ham[pos];
    le = leftenv(cache,pos,mps);
    re = rightenv(cache,pos,mps);

    process_blocks = Map() do (lmask,lblock,e,rblock,rmask)
        cl = le[lmask];
        cr = re[rmask];

        l = transpose(cl[1],(3,1),(2,))*lblock[1];
        for i in 2:length(cl)
            TensorKit.planar_add!(lblock[i],cl[i],true,l,(3,1),(2,));
        end
        nl = transpose(l,(2,3),(1,));

        
        r = rblock[1]*transpose(cr[1],(2,),(1,3));
        for i in 2:length(rblock)
            TensorKit.planar_add!(rblock[i],cr[i],true,r,(2,),(1,3))
        end
        nr = transpose(r,(2,1),(3,));
        

        (nl,e,nr)
    end

    blocks = tcollect(process_blocks,opp.blocks)#,basesize=1);
    
    filter!(blocks) do (l,e,r)
        !(norm(l)<1e-12 || norm(r)<1e-12)
    end
    
    fused_∂∂AC(blocks)
end

function (h::fused_∂∂AC)(x)
    @floop WorkStealingEx() for (l,e,r) in h.blocks
        @init t = similar(x)
        if e isa AbstractTensorMap
            @plansor t[-1 -2;-3] = l[-1 5;4]*x[4 2;1]*e[5 -2;2 3]*r[1 3;-3]
        else
            @plansor t[-1 -2;-3] = l[-1 5;4]*x[4 6;1]*τ[6 5;7 -2]*r[1 7;-3]
            lmul!(e,t);
        end

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
        
        (tl,rblock,rmask)
    end
    
    blocked_left_blocks = tcollect(process_left_blocks,opp1.blocks)#,basesize=1);

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
        
        (lmask,lblock,blocked)
    end

    blocked_right_blocks = tcollect(process_right_blocks,opp2.blocks)#,basesize=1);

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

        
        d = fill(0.0+0im,length(lefts),length(rights));
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

function (h::fused_∂∂AC2)(x)
    
    @floop for (l,temp,r,temp_trans) in h.blocks
        @init temp_3 = fast_similar(x);

        temp_1 = temp();
        mul!(temp_1,l,x)

        temp_2 = temp_trans(temp_1);
        mul!(temp_3,temp_2,r)

        @reduce() do (toret = zero(temp_3); temp_3)
            fast_axpy!(true,temp_3,toret);
        end
    end

    toret
end
