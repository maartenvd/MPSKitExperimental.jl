# these transfers fit in the existing framework - the lower Ab is an excited tensor, the upper one is not.

#transfer, but the upper A is an excited tensor
MPSKit.transfer_left(v::MPSBondTensor, A::MPSTensor, Ab::MPOTensor) =
    @plansor t[-1;-2 -3] := v[1;2]*A[2 3;-3]*conj(Ab[1 3;-2 -1])
MPSKit.transfer_right(v::MPSBondTensor, A::MPSTensor, Ab::MPOTensor) =
    @plansor t[-1;-2 -3] := A[-1 3;1]*v[1;2]*conj(Ab[-3 3;-2 2])

#transfer, but the upper A is an excited tensor and there is an mpo leg being passed through
MPSKit.transfer_left(v::MPSTensor, A::MPSTensor, Ab::MPOTensor) =
    @tensor t[-1 -2;-3 -4] := v[1 -2;2]*A[2 3;-4]*conj(Ab[1 3;-3 -1])
MPSKit.transfer_right(v::MPSTensor, A::MPSTensor, Ab::MPOTensor) =
    @tensor t[-1 -2;-3 -4] := A[-1 3;1]*v[1 -2;2]*conj(Ab[-4 3;-3 2])

#mpo transfer, but with A an excitation-tensor
MPSKit.transfer_left(v::MPSTensor,O::MPOTensor,A::MPSTensor,Ab::MPOTensor) =
    @tensor t[-1 -2;-3 -4] := v[4 5;1]*A[1 3;-4]*O[5 2;3 -2]*conj(Ab[4 2;-3 -1])
MPSKit.transfer_right(v::MPSTensor,O::MPOTensor,A::MPSTensor,Ab::MPOTensor) =
    @tensor t[-1 -2;-3 -4] := A[-1 1;5]*O[-2 3;1 4]*conj(Ab[-4 3;-3 2])*v[5 4;2]

#both A and Ab have an excitation leg
MPSKit.transfer_left(v::MPSTensor,O::MPOTensor,A::MPOTensor,Ab::MPOTensor) =
    @tensor t[-1 -2;-3] := v[4 5;1]*A[1 3;6 -3]*O[5 2;3 -2]*conj(Ab[4 2;6 -1])
MPSKit.transfer_right(v::MPSTensor,O::MPOTensor,A::MPOTensor,Ab::MPOTensor) =
    @tensor t[-1 -2;-3] := A[-1 1;6 5]*O[-2 3;1 4]*conj(Ab[-3 3;6 2])*v[5 4;2]
MPSKit.transfer_left(v::MPSTensor, A::MPOTensor, Ab::MPOTensor) =
    @tensor t[-1 -2;-3] := v[1 -2;2]*A[2 3;4 -3]*conj(Ab[1 3;4 -1])
MPSKit.transfer_right(v::MPSTensor, A::MPOTensor, Ab::MPOTensor) =
    @tensor t[-1 -2;-3] := A[-1 3;4 1]*v[1 -2;2]*conj(Ab[-3 3;4 2])

    #A is an excitation tensor; with an excitation leg
MPSKit.transfer_left(vec::AbstractVector{V},ham::MPSKit.SparseMPOSlice,A::V,Ab::M=A) where V<:MPSTensor where M <:MPOTensor =
MPSKit.transfer_left(M,vec,ham,A,Ab)
MPSKit.transfer_right(vec::AbstractVector{V},ham::MPSKit.SparseMPOSlice,A::V,Ab::M=A) where V<:MPSTensor where M <:MPOTensor =
MPSKit.transfer_right(M,vec,ham,A,Ab)

MPSKit.transfer_left(vec::Vector{V}, ham::MPSKit.SparseMPOSlice, A::O, Ab::O) where {V<:MPSTensor,O<:MPOTensor} = MPSKit.transfer_left(V,vec,ham,A,Ab)
MPSKit.transfer_right(vec::Vector{V}, ham::MPSKit.SparseMPOSlice, A::O, Ab::O) where {V<:MPSTensor,O<:MPOTensor} = MPSKit.transfer_right(V,vec,ham,A,Ab)
