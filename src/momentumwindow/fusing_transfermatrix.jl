struct FusingTransferMatrix{A<:AbstractTensorMap,B<:MPSKit.SparseMPOSlice,C<:AbstractTensorMap,D<:Union{Nothing,AbstractTensorMap}} <: MPSKit.AbstractTransferMatrix
    above::A
    middle::B
    below::C
    fuser::D # if nothing, combine excitation legs. otherwise, use fuser
    isflipped::Bool
end

FusingTransferMatrix(above,middle,below) = FusingTransferMatrix(above,middle,below,nothing,false);

Base.:*(tm::FusingTransferMatrix,v::Union{AbstractTensorMap,AbstractVector}) = tm(v);
Base.:*(v::Union{AbstractTensorMap,AbstractVector},tm::FusingTransferMatrix) = flip(tm)(v);
TensorKit.flip(tm::FusingTransferMatrix) = FusingTransferMatrix(tm.above,tm.middle,tm.below,tm.fuser,!tm.isflipped);

(d::FusingTransferMatrix)(vec) = d.isflipped  ? fusing_transfer_left(vec,d.middle,d.above,d.below,d.fuser) : fusing_transfer_right(vec,d.middle,d.above,d.below,d.fuser);


#A excited, v excited
fusing_transfer_left(v::MPOTensor,O::MPOTensor,A::MPOTensor,Ab::MPSTensor,::Nothing) =
    @plansor t[-1 -2;-3] := v[4 5;6 1]*A[1 3;6 -3]*O[5 2;3 -2]*conj(Ab[4 2;-1])
fusing_transfer_right(v::MPOTensor,O::MPOTensor,A::MPOTensor,Ab::MPSTensor,::Nothing) =
    @plansor t[-1 -2;-3] := A[-1 1;6 5]*O[-2 3;1 4]*conj(Ab[-3 3;2])*v[5 4;6 2]

fusing_transfer_left(v::MPOTensor, A::MPOTensor, Ab::MPSTensor,::Nothing) =
    @plansor t[-1 -2;-3] := v[4 5;6 1]*A[1 3;6 -3]*τ[5 2;3 -2]*conj(Ab[4 2;-1])
fusing_transfer_right(v::MPOTensor, A::MPOTensor, Ab::MPSTensor,::Nothing) =
    @plansor t[-1 -2;-3] := A[-1 1;6 5]*τ[-2 3;1 4]*conj(Ab[-3 3;2])*v[5 4;6 2]

#Ab excited, v excited
fusing_transfer_left(v::MPOTensor,O::MPOTensor,A::MPSTensor,Ab::MPOTensor,::Nothing) =
    @plansor t[-1 -2;-3] := v[4 5;6 1]*A[1 3;-3]*O[5 2;3 -2]*conj(Ab[4 2;6 -1])
fusing_transfer_right(v::MPOTensor,O::MPOTensor,A::MPSTensor,Ab::MPOTensor,::Nothing) =
    @plansor t[-1 -2;-3] := A[-1 1;5]*O[-2 3;1 4]*conj(Ab[-3 3;6 2])*v[5 4;6 2]
fusing_transfer_left(v::MPOTensor, A::MPSTensor, Ab::MPOTensor,::Nothing) =
    @tensor t[-1 -2;-3] := v[1 -2;4 2]*A[2 3;-3]*conj(Ab[1 3;4 -1])
fusing_transfer_right(v::MPOTensor, A::MPSTensor, Ab::MPOTensor,::Nothing) =
    @tensor t[-1 -2;-3] := A[-1 3;1]*v[1 -2;4 2]*conj(Ab[-3 3;4 2])

#A excited, Ab excited
fusing_transfer_left(v::MPSTensor,O::MPOTensor,A::MPOTensor,Ab::MPOTensor,::Nothing) =
    @plansor t[-1 -2;-3] := v[4 5;1]*A[1 3;6 -3]*O[5 2;3 -2]*conj(Ab[4 2;6 -1])
fusing_transfer_right(v::MPSTensor,O::MPOTensor,A::MPOTensor,Ab::MPOTensor,::Nothing) =
    @plansor t[-1 -2;-3] := A[-1 1;6 5]*O[-2 3;1 4]*conj(Ab[-3 3;6 2])*v[5 4;2]
fusing_transfer_left(v::MPSTensor, A::MPOTensor, Ab::MPOTensor,::Nothing) =
    @tensor t[-1 -2;-3] := v[1 -2;2]*A[2 3;4 -3]*conj(Ab[1 3;4 -1])
fusing_transfer_right(v::MPSTensor, A::MPOTensor, Ab::MPOTensor,::Nothing) =
    @tensor t[-1 -2;-3] := A[-1 3;4 1]*v[1 -2;2]*conj(Ab[-3 3;4 2])


fusing_transfer_left(vec::AbstractVector{V}, ham::MPSKit.SparseMPOSlice, A::O, Ab::O,::Nothing) where {V<:MPSTensor,O<:MPOTensor} = fusing_transfer_left(V,vec,ham,A,Ab,nothing)
fusing_transfer_right(vec::AbstractVector{V}, ham::MPSKit.SparseMPOSlice, A::O, Ab::O,::Nothing) where {V<:MPSTensor,O<:MPOTensor} = fusing_transfer_right(V,vec,ham,A,Ab,nothing)
fusing_transfer_left(vec::AbstractVector{O}, ham::MPSKit.SparseMPOSlice, A::O, Ab::V,::Nothing) where {V<:MPSTensor,O<:MPOTensor} = fusing_transfer_left(V,vec,ham,A,Ab,nothing)
fusing_transfer_right(vec::AbstractVector{O}, ham::MPSKit.SparseMPOSlice, A::O, Ab::V,::Nothing) where {V<:MPSTensor,O<:MPOTensor} = fusing_transfer_right(V,vec,ham,A,Ab,nothing)
fusing_transfer_left(vec::AbstractVector{O}, ham::MPSKit.SparseMPOSlice, A::V, Ab::O,::Nothing) where {V<:MPSTensor,O<:MPOTensor} = fusing_transfer_left(V,vec,ham,A,Ab,nothing)
fusing_transfer_right(vec::AbstractVector{O}, ham::MPSKit.SparseMPOSlice, A::V, Ab::O,::Nothing) where {V<:MPSTensor,O<:MPOTensor} = fusing_transfer_right(V,vec,ham,A,Ab,nothing)


function fusing_transfer_left(RetType,vec,ham::MPSKit.SparseMPOSlice,A,Ab,fuser)
    toret = similar(vec,RetType,length(vec));

    @threads for k in 1:length(vec)

        els = keys(ham,:,k);


        @floop WorkStealingEx() for j in els
            if ham.Os[j,k] isa Number
                t = lmul!(ham.Os[j,k], fusing_transfer_left(vec[j],A,Ab,fuser))
            else
                t = fusing_transfer_left(vec[j],ham[j,k],A,Ab,fuser)
            end

            @reduce(s = MPSKit.inplace_add!(nothing,t))
        end

        if isnothing(s)
            s = fusing_transfer_left(vec[1],ham[1,k],A,Ab,fuser)
        end
        toret[k] = s;
    end

    toret
end
function fusing_transfer_right(RetType,vec,ham::MPSKit.SparseMPOSlice,A,Ab,fuser)
    toret = similar(vec,RetType,length(vec));

    @threads for j in 1:length(vec)

        els = keys(ham,j,:)

        @floop WorkStealingEx() for k in els
            if ham.Os[j,k] isa Number
                t = lmul!(ham.Os[j,k],fusing_transfer_right(vec[k],A,Ab,fuser))
            else 
                t = fusing_transfer_right(vec[k],ham[j,k],A,Ab,fuser)
            end

            @reduce(s = MPSKit.inplace_add!(nothing,t))
        end

        if isnothing(s)
            s = fusing_transfer_right(vec[1],ham[j,1],A,Ab,fuser)
        end

        toret[j] = s
    end

    toret
end
