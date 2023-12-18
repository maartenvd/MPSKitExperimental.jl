const rawdat = matread("haagerup/H3_data.mat");
const qdim_dat = rawdat["qdim"];
const nsymb_dat = rawdat["Nsymbol"];
const fsymb_dat = rawdat["Fsymbol"];

struct H3 <: TensorKit.Sector
    i::Int
end

Base.one(::Type{H3}) = H3(1); # beetje een assumptie
Base.conj(s::H3) = H3(findfirst(nsymb_dat[s.i,:,1] .== 1)) #kweetnie?
Base.isreal(::Type{H3}) = false #kweet nie of da zo is, of zelfs nodig is (tis zo voor ising)
Base.isless(a::H3, b::H3) = a.i < b.i
Base.convert(::Type{H3},i::Int) = H3(i);

Base.length(::TensorKit.SectorValues{H3}) = length(qdim_dat);
Base.iterate(iter::TensorKit.SectorValues{H3},state = 1) = state > length(iter) ? nothing : (state,state+1);
TensorKit.findindex(::TensorKit.SectorValues{H3},v::H3) = v.i;
Base.getindex(::TensorKit.SectorValues{H3},i) = H3(i);

TensorKit.FusionStyle(::Type{H3}) = SimpleFusion();
TensorKit.BraidingStyle(::Type{H3}) = TensorKit.NoBraiding();
TensorKit.dim(d::H3) = qdim_dat[d.i];

TensorKit.:⊗(a::H3, b::H3) = H3.(findall(x->x>0,nsymb_dat[a.i,b.i,:]));
TensorKit.Nsymbol(a::H3,b::H3,c::H3) = nsymb_dat[a.i,b.i,c.i];
TensorKit.Fsymbol(a::H3, b::H3, c::H3,d::H3, e::H3, f::H3) = fsymb_dat[a.i,b.i,c.i,d.i,e.i,f.i];
