
struct SymbolicTensorMap{A,B}
    structure::B
    SymbolicTensorMap(a,b) = new{a,typeof(b)}(b)
end
ttype(d::SymbolicTensorMap{A,B}) where {A,B} = A

TensorOperations.tensorfree!(::SymbolicTensorMap, ::TensorOperations.Backend) = nothing
TensorOperations.scalartype(a::SymbolicTensorMap) = TensorOperations.scalartype(ttype(a))
TensorOperations.tensorscalar(a::SymbolicTensorMap) = zero(TensorOperations.scalartype(a))

TensorKit.sectortype(a::SymbolicTensorMap) = sectortype(typeof(a))
TensorKit.sectortype(::Type{SymbolicTensorMap{A,B}}) where {A,B} = sectortype(B)

TensorKit.numin(a::SymbolicTensorMap) = length(domain(a.structure))
TensorKit.numout(a::SymbolicTensorMap) = length(codomain(a.structure))

TensorKit.codomainind(a::SymbolicTensorMap) = ntuple(x->x,numout(a))
TensorKit.domainind(a::SymbolicTensorMap) = ntuple(x->x+numout(a),numin(a))