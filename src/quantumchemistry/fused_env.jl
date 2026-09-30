#=
    environments for a FusedMPOHamiltonian always live on disk
=#
MPSKit.environments(state::FiniteMPS,ham::FusedMPOHamiltonian) = disk_environments(state,ham)
function MPSKit.environments(below::FiniteMPS,ham::FusedMPOHamiltonian,above::FiniteMPS)
    below === above || throw(ArgumentError("FusedMPOHamiltonian environments require below === above"))
    return disk_environments(below,ham)
end

#=
    need to define the relevant transfer operators
=#

mpotype(O::FusedSparseBlock{E,H,Sp}) where {E,H,Sp} = H

function MPSKit.transfer_left(v::Vector,O::FusedSparseBlock,A,Ab=A)
    Ab_flipped = convert(TensorMap,transpose(Ab',((1,3),(2,))));
    mapper = Map() do (lmask,lblock,e,rblock,rmask)
        # reduce left
        v_masked = v[lmask];

        l = rmul!(copy(v_masked[1]),lblock[1]);
        for i in 2:length(v_masked)
            l = axpy!(lblock[i],v_masked[i],l);
        end

        @planar allocator = malloc() y[-1 -2;-3] := l[4 2;1]*A[1 3;-3]*e[2 5;3 -2]*Ab_flipped[-1 5;4]

        # expand r
        toret = map(rblock) do r
            (r,y)
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
                out[i] = axpy!(ct,cnr,out[i])
            else
                out[i] = rmul!(copy(cnr),ct)
                isassigned[i] = true
            end
        end
    end

    for i in findall(!,isassigned)
        out[i] = zeros(eltype(A),space(Ab,3)'*O.imspaces[i]←space(A,3)')
    end

    out
end


@tightloop_planar right_trans allocator = malloc() y[-1 -2;-3] := A[-1 2;1]*v[1 3;4]*O[-2 5;2 3]*Ab[4 5;-3]
function MPSKit.transfer_right(v::Vector,O::FusedSparseBlock,A,Ab=A)
    # first we should do a pass through O/v for the factories, which can then be utilized in parallel
    Ab_flipped = transpose(Ab',((1,3),(2,)));

    mapper = Map() do (lmask,lblock,e,rblock,rmask)
        v_masked = v[rmask];

        r = rmul!(copy(v_masked[1]),rblock[1]);
        for i in 2:length(rblock)
            r = axpy!(rblock[i],v_masked[i],r)
        end
        
        @planar allocator = malloc() nr[-1 -2;-3] := A[-1 2;1]*r[1 3;4]*e[-2 5;2 3]*Ab_flipped[4 5;-3]

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
                out[i] = axpy!(ct,cnr,out[i])
            else
                out[i] = rmul!(copy(cnr),ct)
                isassigned[i] = true
            end
        end
    end

    for i in findall(!,isassigned)
        out[i] = zeros(eltype(A),space(A,1)*O.domspaces[i]←space(Ab,1))
    end
    out
end
