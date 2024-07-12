# ac_prime

struct Tight_AC_prime{T}
    table::T
end



#@overlay tighthacktable function MPSKit.∂∂AC(pos::Int, mps, mpoham::Union{MPOHamiltonian,SparseMPO}, cache)
function tighthack(::typeof(MPSKit.∂∂AC),pos::Int, mps, mpoham::Union{MPOHamiltonian,SparseMPO}, cache)
    le = leftenv(cache,pos,mps)
    re = rightenv(cache,pos,mps)

    ac_type = typeof(mps.AC[pos])
    ac_structure = space(mps.AC[pos])

    table = map(keys(mpoham[pos])) do (i,j)
        @planar lblock[-1 -2 -3;-4 -5] := le[i][-1 1;-4]*mpoham[pos][i,j][1 -2;-5 -3]
        rblock = re[j]
        
        contract = tight_ac_contractor(left=(typeof(lblock),space(lblock)),x = (ac_type,ac_structure), right = (typeof(rblock),space(rblock)), out = (ac_type,ac_structure))

        (lblock,rblock,contract)
    end

    return Tight_AC_prime(table)
end


(h::Tight_AC_prime)(x) = _reduce_tight_ac(h.table,x,ceil(length(h.table)/nthreads()))
function _reduce_tight_ac(blocks,x,basesize)
    if length(blocks) <= basesize
        toret = zero(x)

        for (l,r,factory) in blocks
            factory(left=l,right=r,x=x,out=toret)
        end

        return toret
    else
        spl = Int(ceil(length(blocks)/2));
        t = @Threads.spawn _reduce_tight_ac(blocks[1:spl],x,basesize)
        toret = _reduce_tight_ac(view(blocks,spl+1:length(blocks)),x,basesize)
        fast_axpy!(true,fetch(t),toret)
        return toret
    end
end


# c_prime
#@overlay tighthacktable function MPSKit.∂∂C(pos::Int, mps, mpoham::Union{MPOHamiltonian,SparseMPO}, cache) 
function tighthack(::typeof(MPSKit.∂∂C),pos::Int, mps, mpoham::Union{MPOHamiltonian,SparseMPO}, cache) 

    le = leftenv(cache,pos+1,mps)
    re = rightenv(cache,pos,mps)

    c_type = typeof(mps.CR[pos])
    c_structure = space(mps.CR[pos])

    table = map(zip(le,re)) do (lblock,rblock)    
        contract = tight_c_contractor(left=(typeof(lblock),space(lblock)),x = (c_type,c_structure), right = (typeof(rblock),space(rblock)), out = (c_type,c_structure))

        (lblock,rblock,contract)
    end

    return Tight_AC_prime(table)
end


Base.:*(h::Union{Tight_AC_prime}, v) = h(v);

#--------------------------------------------------------------------------------------

# regularized transfermatrix with 3 special case contractions optimized - the case where we contract with a bond tensor and the case where we contract with a tensor with trivial charge (oneunit or oneunit')

struct FastRegTransferMatrix{T,L,R,F1,F2,F3} <: MPSKit.AbstractTransferMatrix
    tm::T
    lvec::L
    rvec::R
    f1::F1
    f2::F2
    f3::F3
end


function (tm::FastRegTransferMatrix)(a)
    out = tm.tm(a)
    if a isa MPSBondTensor
        tm.f1(v = out,lvec = tm.lvec, rvec = tm.rvec)
    elseif a isa MPSTensor && space(a,2) == oneunit(space(a,1))
        tm.f2(v = out,lvec = tm.lvec, rvec = tm.rvec)
    elseif a isa MPSTensor && space(a,2) == oneunit(space(a,1))'
        tm.f3(v = out,lvec = tm.lvec, rvec = tm.rvec)
    else
        MPSKit.regularize!(out,tm.lvec,tm.rvec)
    end

    return out
end

MPSKit.flip(a::FastRegTransferMatrix) = MPSKit.regularize(flip(a.tm),a.rvec,a.lvec)

function tighthack(::typeof(MPSKit.regularize), t::MPSKit.AbstractTransferMatrix, lvec, rvec)
    if lvec isa MPSBondTensor
        # compile two special cases for this transfermatrix:
        f1 = fast_reg_bond(v = (typeof(rvec),space(rvec)),lvec = (typeof(lvec),space(lvec)),rvec = (typeof(rvec),space(rvec)))
        
        l_mps_type = tensormaptype(spacetype(lvec),2,1,storagetype(lvec))
        l_space = space(rvec,1)*oneunit(space(rvec,1))←space(rvec,2)'
        f2 = fast_reg_mps(v = (l_mps_type,l_space),lvec = (typeof(lvec),space(lvec)),rvec = (typeof(rvec),space(rvec)))

        l_space = space(rvec,1)*oneunit(space(rvec,1))'←space(rvec,2)'
        f3 = fast_reg_mps(v = (l_mps_type,l_space),lvec = (typeof(lvec),space(lvec)),rvec = (typeof(rvec),space(rvec)))
        return FastRegTransferMatrix(t,lvec,rvec,f1,f2,f3)
    else
        return MPSKit.RegTransferMatrix(t, lvec, rvec);
    end
end

#--------------------------------------------------------------------------------------
# regular MPS transfermatrix, again with 3 special case contractions optimized
struct FastSingleTransferMatrix{A<:MPSTensor,C,F1L,F1R,F2L,F2R,F3L,F3R,B1,B2} <:
    MPSKit.AbstractTransferMatrix
 above::A
 below::C
 isflipped::Bool
 
 f1_left::F1L
 f1_right::F1R

 f2_left::F2L
 f2_right::F2R
 
 f3_left::F3L
 f3_right::F3R

    braid1::B1
    braid2::B2
end

MPSKit.flip(d::FastSingleTransferMatrix) = FastSingleTransferMatrix(d.above,d.below,!d.isflipped,d.f1_left,d.f1_right,d.f2_left,d.f2_right,d.f3_left,d.f3_right,d.braid1,d.braid2)
function (d::FastSingleTransferMatrix)(vec)
    if vec isa MPSBondTensor && d.isflipped
        return d.f1_left(v=vec,a=d.above,b=d.below)
    elseif vec isa MPSBondTensor
        return d.f1_right(v=vec,a=d.above,b=d.below)
    elseif vec isa MPSTensor && d.isflipped
        if space(vec,2) == oneunit(space(vec,2))
            return d.f2_left(v = vec, a = d.above, b = d.below, braid = d.braid1)
        elseif space(vec,2) == oneunit(space(vec,2))'
            return d.f3_left(v = vec, a = d.above, b = d.below, braid = d.braid2)
        else
            return MPSKit.transfer_left(vec,d.above,copy(d.below'))
        end
    elseif vec isa MPSTensor
        if space(vec,2) == oneunit(space(vec,2))
            return d.f2_right(v = vec, a = d.above, b = d.below, braid = d.braid2)
        elseif space(vec,2) == oneunit(space(vec,2))'
            return d.f3_right(v = vec, a = d.above, b = d.below, braid = d.braid1)
        else
            return MPSKit.transfer_right(vec,d.above,copy(d.below'))
        end
    elseif d.isflipped
        return MPSKit.transfer_left(vec,d.above,copy(d.below')) 
    else
        return MPSKit.transfer_right(vec,d.above,copy(d.below'))
    end
end;

function tighthack(::typeof(MPSKit.TransferMatrix),a::AbstractTensorMap, b, oc::AbstractTensorMap, isflipped=false)
    if isnothing(b) && a isa MPSTensor
        c = copy(oc') # my tighthack_planar code cannot deal with adjoint (how would I define the adjoint of a SymbolicTensorMap?)
        bond_type = tensormaptype(spacetype(a),1,1,storagetype(a))
        
        f1_left = fast_left_bond(v = (bond_type,space(oc,1)←space(a,1)), a = (typeof(a),space(a)), b = (typeof(c),space(c)))
        f1_right = fast_right_bond( a = (typeof(a),space(a)), b = (typeof(c),space(c)), v = (bond_type, space(a,3)'←space(oc,3)'))

        triv_braid = copy(TensorKit.BraidingTensor(space(a, 2), dual(oneunit(space(a,2)))))
        trivp_braid = copy(TensorKit.BraidingTensor(space(a, 2), oneunit(space(a,2))))


        f2_left = fast_left_mps(v = (typeof(a),space(c,1)*oneunit(space(a,1))←space(a,1)),braid=(typeof(triv_braid),space(triv_braid)), a = (typeof(a),space(a)), b = (typeof(c),space(c)))
        f2_right = fast_right_mps( a = (typeof(a),space(a)), b = (typeof(c),space(c)), v = (typeof(a), space(a,3)'*oneunit(space(a,1))←space(oc,3)'),braid=(typeof(trivp_braid),space(trivp_braid)))

        f3_left = fast_left_mps(v = (typeof(a),space(c,1)*oneunit(space(a,1))'←space(a,1)),braid=(typeof(trivp_braid),space(trivp_braid)), a = (typeof(a),space(a)), b = (typeof(c),space(c)))
        f3_right = fast_right_mps(a = (typeof(a),space(a)), b = (typeof(c),space(c)), v = (typeof(a), space(a,3)'*oneunit(space(a,1))'←space(oc,3)'),braid=(typeof(triv_braid),space(triv_braid)))
        
        return FastSingleTransferMatrix(a,c,isflipped,f1_left,f1_right,f2_left,f2_right,f3_left,f3_right,triv_braid,trivp_braid)

    else
        return MPSKit.SingleTransferMatrix(a,b,oc,isflipped)
    end
end