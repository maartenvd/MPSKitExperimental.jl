#%% definition of leftgauged window with a momentum
using MPSKit:MPSTensor,MPSBondTensor,MPOTensor,utilleg,_firstspace,_lastspace,_transpose_tail,_transpose_front,Multiline,LeftGaugedQP;

#=
Momentum superposition of different windows.
Every momentum window starts with a VL tensor (gaugefixing).

LeftGaugedMW.AL[row,col] then physically corresponds with something at location row+col
AL[1,1] is the first AL tensor following the VL tensor - which sits at location 1 - and therefore sits physcially at location 2
=#

struct LeftGaugedMW{O<:MPOTensor,F<:FiniteMPS,G<:InfiniteMPS}
    VLs :: PeriodicArray{O,1}

    variational :: Multiline{F}
    momentum :: ComplexF64

    left_gs :: G
    right_gs :: G
end
Base.size(st::LeftGaugedMW) = size(st.variational);
Base.size(st::LeftGaugedMW,i) = size(st)[i];
Base.copy(st::LeftGaugedMW) = LeftGaugedMW(copy(st.VLs),copy(st.variational),st.momentum,st.left_gs,st.right_gs);
MPSKit.utilleg(st::LeftGaugedMW) = space(st.VLs[1],3);

function LeftGaugedMW(datfun, len::Int, maxvirtspace, left_gs::InfiniteMPS, right_gs::InfiniteMPS=left_gs; utilspace = oneunit(virtualspace(left_gs,1)), momentum = 0.0)
    length(left_gs) == length(right_gs) || throw(ArgumentError("period mismatch"));

    VLs = map(left_gs.AL) do al
        vl = convert(TensorMap,adjoint(rightnull(adjoint(al))));
        utl = isomorphism(storagetype(vl),fuse(utilspace*space(vl,3)'),utilspace*space(vl,3)')
        @plansor VL[-1 -2;-3 -4] := vl[-1 -2;1]*conj(utl[-4;-3 1])
    end

    variational = Multiline(map(1:length(left_gs)) do row
        physical_spaces = [space(left_gs,row+col) for col in 1:len];
        FiniteMPS(datfun,eltype(VLs[row]),physical_spaces,maxvirtspace,left=_lastspace(VLs[row])',right=virtualspace(right_gs,row+len))
    end)

    LeftGaugedMW(VLs,variational,momentum,left_gs,right_gs)
end

function Base.:*(a::Number,b::LeftGaugedMW)
    b = copy(b)
    for row in 1:length(b.variational)
        b.AC[row,1]*=a;
    end
    b
end

function TensorKit.lmul!(s::MPSBondTensor,st::LeftGaugedMW)
    # instead of modifying VL, we can "pull this through" to the c tensor
    for row in 1:size(st,1)
        @plansor t[-1;-2] := conj(st.VLs[row][1 2;4 -1])*s[4;3]*st.VLs[row][1 2;3 -2]
        #=
        @tensor w1[-1 -2;-3 -4] := st.VLs[row][-1 -2;1 -4]*s[-3;1];
        @tensor w2[-1 -2;-3 -4] := st.VLs[row][-1 -2;-3 1]*t[1;-4]
        @show norm(w1-w2)
        =#
        st.CR[row,0] = t*st.CR[row,0]
    end

    st
end

function Base.getproperty(st::LeftGaugedMW,s::Symbol)
    if s == :trivial
        return st.left_gs === st.right_gs
    elseif s in (:AL,:AR,:AC,:CR)
        return Base.getproperty(st.variational,s);
    else
        return getfield(st,s);
    end
end

function TensorKit.normalize!(st::LeftGaugedMW)
    normalize!.(st.variational.data)
    st
end

function extend(st::LeftGaugedMW,amount::Int)
    LeftGaugedMW(st.VLs,Multiline(map(enumerate(st.variational.data)) do (i,row)
        ALs = copy(row.ALs);
        ARs = copy(row.ARs);
        ACs = copy(row.ACs);
        CLs = copy(row.CLs);
        append!(ALs,fill(missing,amount));
        append!(ACs,fill(missing,amount));
        append!(CLs,fill(missing,amount));
        append!(ARs,st.right_gs.AR[size(st,2)+i+1:size(st,2)+i+amount])
        #@show length(ALs),length(ARs),length(ACs),length(CLs)
        FiniteMPS(ALs,ARs,ACs,CLs)
    end),st.momentum,st.left_gs,st.right_gs);
end

function TensorKit.dot(a::LeftGaugedMW,b::LeftGaugedMW)
    a.left_gs == b.left_gs && a.right_gs == b.right_gs && size(a,1) == size(b,1) || throw(ArgumentError("nonsensical inproduct"))

    if size(a,2) > size(b,2)
        b = extend(b,size(a,2)-size(b,2));
    elseif size(b,2) > size(a,2)
        a = extend(a,size(b,2)-size(a,2));
    end

    sum(map(1:size(a,1)) do row
        @tensor v[-1;-2] := conj(a.VLs[row][1,2,3,-1])*b.VLs[row][1,2,3,-2]
        v = v * TransferMatrix(b.AL[row,:],a.AL[row,:]);
        tr(adjoint(a.CR[row,end])*v*b.CR[row,end])
    end)
end

TensorKit.norm(a::LeftGaugedMW) = real(sqrt(dot(a,a)))

function partialdot(a::LeftGaugedMW,b::LeftGaugedMW)
    a.left_gs == b.left_gs && a.right_gs == b.right_gs && size(a,1) == size(b,1) || throw(ArgumentError("nonsensical inproduct"))

    if size(a,2) > size(b,2)
        b = extend(b,size(a,2)-size(b,2));
    elseif size(b,2) > size(a,2)
        a = extend(a,size(b,2)-size(a,2));
    end

    sum(map(1:size(a,1)) do row
        t = b.CR[row,end]*a.CR[row,end]'
        t = TransferMatrix(b.AL[row,:],a.AL[row,:])*t;
        @tensor s[-1;-2] := b.VLs[row][3 4;-1 1]*t[1;2]*conj(a.VLs[row][3 4;-2 2])
    end)
end

function projdown(row,col,a,b,s=isomorphism(utilleg(b),utilleg(a)))
    if size(a,2) > size(b,2)
        b = extend(b,size(a,2)-size(b,2));
    elseif size(b,2) > size(a,2)
        a = extend(a,size(b,2)-size(a,2));
    end

    @tensor lstart[-1;-2] := a.VLs[row][3,2,1,-2]*s[4,1]*conj(b.VLs[row][3,2,4,-1]);
    lstart = lstart* TransferMatrix(a.AL[row,1:col-1],b.AL[row,1:col-1]);
    rstart = TransferMatrix(a.AR[row,col+1:end],b.AR[row,col+1:end]) * one(a.CR[row,end]);
    @tensor y[-1 -2;-3] := lstart[-1;1]*a.AC[row,col][1 -2;2]*rstart[2;-3]
end

function Base.convert(::Type{<:LeftGaugedMW},a::LeftGaugedQP)
    left_gs = a.left_gs;
    right_gs = a.right_gs;
    utilspace = utilleg(a);

    bundle = map(zip(a.VLs,a.Xs)) do (vl,x)
        utl = isomorphism(storagetype(vl),utilspace*space(vl,3)',fuse(utilspace*space(vl,3)'))
        @plansor VL[-1 -2;-3 -4] := vl[-1 -2;1]*utl[-3 1;-4]
        @plansor C[-1;-2] := conj(utl[2 1;-1])*x[1;2 -2]
        (VL,C)
    end
    VLs = PeriodicArray(first.(bundle))

    variational = Multiline(map(enumerate(last.(bundle))) do (i,c)
        @tensor ac[-1 -2;-3] := c[-1;1]*right_gs.AR[i+1][1 -2;-3]
        FiniteMPS([ac]);
    end)

    LeftGaugedMW(VLs,variational,a.momentum,left_gs,right_gs)
end
