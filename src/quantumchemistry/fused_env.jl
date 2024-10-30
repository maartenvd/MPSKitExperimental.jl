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
        @assert false
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


@tightloop_planar right_trans y[-1 -2;-3] := A[-1 2;1]*v[1 3;4]*O[-2 5;2 3]*Ab[4 5;-3]
function MPSKit.transfer_right(v::Vector,O::FusedSparseBlock,A,Ab=A)
    # first we should do a pass through O/v for the factories, which can then be utilized in parallel
    Ab_flipped = transpose(Ab',(1,3),(2,));

    homspace_example = O.pspace*O.pspace←O.pspace*O.pspace
    homspace_type = typeof(homspace_example)
    factory_type = typeof(right_trans(v=(typeof(v[1]),space(v[1])),A = (typeof(A),space(A)),Ab = (typeof(Ab_flipped),space(Ab_flipped)), O = (mpotype(O),homspace_example)))
    factories = Dict{homspace_type,factory_type}();
    promise_creation = Dict{homspace_type,Any}();


    for (lmask,lblock,e,rblock,rmask) in O.blocks
        e isa Number && @assert false #not supported anymore
        
        space(e) in keys(promise_creation) && continue #someone else will make this element
        promise_creation[space(e)] = @Threads.spawn begin
            v_example = v[findfirst(rmask)]
            right_trans(v = (typeof(v_example),space(v_example)),A = (typeof(A),space(A)),Ab = (typeof(Ab_flipped),space(Ab_flipped)), O = (typeof(e),space(e)))
   
        end
    end

    for (k,v) in promise_creation
        factories[k] = fetch(v)
    end

    mapper = Map() do (lmask,lblock,e,rblock,rmask)
        v_masked = v[rmask];

        r = rmul!(fast_copy(v_masked[1]),rblock[1]);
        for i in 2:length(rblock)
            r = fast_axpy!(rblock[i],v_masked[i],r)
        end
        
        nr = factories[space(e)](v = r, A = A, Ab = Ab_flipped,O = e)

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
        @assert false
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
