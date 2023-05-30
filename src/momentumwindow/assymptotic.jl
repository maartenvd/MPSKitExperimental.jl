
struct AssymptoticScatter{A,B}#{A<:MPSKit.LeftGaugedQP,B<:MPSKit.RightGaugedQP}
    B1::A
    B2::B
end

function Base.getproperty(st::AssymptoticScatter,s::Symbol)
    if s == :momentum
        return st.B1.momentum + st.B2.momentum
    elseif s == :trivial
        return st.B1.trivial && st.B2.trivial
    else
        return getfield(st,s)
    end
end

function MPSKit.utilleg(st::AssymptoticScatter)
    u1 = utilleg(st.B1);
    u2 = utilleg(st.B2);
    fuse(u1*u2)
end

#%%

TensorKit.dot(a::AssymptoticScatter,b::LeftGaugedMW) = dot(b,a)';
TensorKit.dot(init::LeftGaugedMW,scatter::AssymptoticScatter) = tr(partialdot(init,scatter))

partialdot(scatter::AssymptoticScatter,init::LeftGaugedMW) = partialdot(init,scatter)'
function partialdot(init::LeftGaugedMW,scatter::AssymptoticScatter)
    (scatter.B1.left_gs == init.left_gs && scatter.B2.right_gs == init.right_gs) || throw(ArgumentError("I mean ...")) # nonsensical inputs
    scatter.momentum == init.momentum || throw(ArgumentError("momentum should match"))
    #utilleg(scatter) == utilleg(init) || throw(ArgumentError("util leg space mismatch"))

    fuser = isomorphism(utilleg(init),utilleg(scatter.B2)*utilleg(scatter.B1))
    sum(map(1:size(init,1)) do row
        len = size(init,2);
        start = init.CR[row,len]'
        A_start = TensorMap(zeros,_firstspace(start),utilleg(scatter.B2)'*_lastspace(start)');
        
        for i in size(init,2):-1:1
            A_start = TransferMatrix(scatter.B2.left_gs.AL[row+i],init.AL[row,i]) * A_start;
            @plansor  A_start[-1;-2 -3] += (exp(1im*scatter.B2.momentum*i)*scatter.B2[row+i])[-1 2;-2 1]*start[1;3]*conj(init.AL[row,i][-3 2;3])

            start = TransferMatrix(init.right_gs.AR[row+i],init.AL[row,i]) * start;
        end

        @plansor  ρ[-1;-2] := scatter.B1[row][3 4;6 1]*inv(scatter.B1.right_gs.CR[row])[1;2]*A_start[2;7 5]*conj(init.VLs[row][3 4;-2 5])*fuser[-1;7 6]
    end)
end


#%%

#=
keeps track of the overlap
implicitly we do H-E - not just H
=#

struct ScatterOverlap{A,B,D,E} <: Cache
    ldependencies::Matrix{A} #the data we used to calculate leftenvs/rightenvs
    rdependencies::Matrix{A}

    opp::B #the operator
    E::Float64 # the energy of the scatterstate, which we have to subtract

    leftenvs::Matrix{Vector{E}}
    rightenvs::Matrix{Vector{A}}

    Aleftenvs::Matrix{Vector{E}}
    Brightenvs::Matrix{Vector{E}}

    ABleftenvs::Matrix{Vector{A}}
    ABrightenvs::Matrix{Vector{E}}

    above::D
end

function MPSKit.environments(init::LeftGaugedMW,toapprox::Tuple{<:MPOHamiltonian,<:AssymptoticScatter})
    (ham,scatter) = toapprox;

    left_gs = scatter.B1.left_gs;
    middle_gs = scatter.B1.right_gs;
    right_gs = scatter.B2.right_gs;

    scatter.trivial || throw(ArgumentError("for now we only support trivial excitations")) # may get lifted
    (scatter.B1.left_gs == init.left_gs && init.trivial) || throw(ArgumentError("I mean ...")) # nonsensical inputs
    scatter.momentum == init.momentum || throw(ArgumentError("momentum should match"))
    #utilleg(scatter) == utilleg(init) || throw(ArgumentError("util leg space mismatch"))

    fuser = isomorphism(utilleg(init),utilleg(scatter.B2)*utilleg(scatter.B1));

    K1 = scatter.B1.momentum;
    K2 = scatter.B2.momentum;
    K = K1+K2;
    envs = environments(left_gs,ham);
    Aenvs = environments(scatter.B1,ham,envs);
    Benvs = environments(scatter.B2,ham,envs);
    
    lAB = left_double_env(K1+K2,scatter.B2,Aenvs.lBs,ham,fuser);
    rAB = right_double_env(K1+K2,scatter.B1,Benvs.rBs,ham,fuser);

    E = real(dot(scatter.B1,effective_excitation_hamiltonian(ham,scatter.B1,Aenvs)) + dot(scatter.B2,effective_excitation_hamiltonian(ham,scatter.B2,Benvs)));
    E += real(expectation_value(left_gs,ham,size(init,2)+1));

    O_type = eltype(ham[1]);
    A_type = tensormaptype(spacetype(O_type),2,1,storagetype(O_type));
    E_type = tensormaptype(spacetype(O_type),2,2,storagetype(O_type));

    #fill in all fields
    ldependencies = fill(similar(init.AL[1,1]),size(init,1),size(init,2));
    rdependencies = fill(similar(init.AL[1,1]),size(init,1),size(init,2));

    leftenvs = Matrix{Vector{E_type}}(undef,size(init,1),size(init,2)+1);
    rightenvs = Matrix{Vector{A_type}}(undef,size(init,1),size(init,2)+1);

    Aleftenvs = Matrix{Vector{E_type}}(undef,size(init,1),size(init,2)+1);
    Brightenvs = Matrix{Vector{E_type}}(undef,size(init,1),size(init,2)+1);
    
    ABleftenvs = Matrix{Vector{A_type}}(undef,size(init,1),size(init,2)+1);
    ABrightenvs = Matrix{Vector{E_type}}(undef,size(init,1),size(init,2)+1);

    for row in 1:size(init,1)
        leftenvs[row,1] = leftenv(envs,row,left_gs)*TransferMatrix(left_gs.AL[row],ham[row],init.VLs[row]);
        rightenvs[row,end] = rightenv(envs,row+size(init,2),left_gs);
        
        Aleftenvs[row,1] =  map(1:ham.odim) do k
            sum(map(1:ham.odim) do j
                @plansor out[-1 -2;-3 -4]:= Aenvs.lBs[j,row][1 2;3 4]*conj(init.VLs[row][1 5;6 -1])*fuser[6;-3 3]*ham[row][j,k][2 5;7 -2]*middle_gs.AR[row][4 7;-4]
                @plansor out[-1 -2;-3 -4]+= leftenv(envs,row,left_gs)[j][1 2;4]*conj(init.VLs[row][1 5;6 -1])*fuser[6;-3 3]*ham[row][j,k][2 5;7 -2]*scatter.B1[row][4 7;3 -4]
            end)
        end

        ABleftenvs[row,1] = map(1:ham.odim) do k
            sum(map(1:ham.odim) do j
                @plansor out[-1 -2;-3]:= lAB[j,row][1 2;3 4]*conj(init.VLs[row][1 5;3 -1])*ham[row][j,k][2 5;7 -2]*right_gs.AR[row][4 7;-3]
                @plansor out[-1 -2;-3]+= Aenvs.lBs[j,row][1 2;3 4]*inv(middle_gs.CR[row-1])[4;5]*conj(init.VLs[row][1 6;7 -1])*fuser[7;8 3]*ham[row][j,k][2 6;9 -2]*scatter.B2[row][5 9;8 -3]
            end)
        end

        Brightenvs[row,end] = (Benvs.rBs.*exp(1im*K2*size(init,2)))[:,row+size(init,2)]
        ABrightenvs[row,end] = (rAB.*exp(1im*K*size(init,2)))[:,row+size(init,2)];
    
    end

    ScatterOverlap(ldependencies,rdependencies,ham,E,leftenvs,rightenvs,Aleftenvs,Brightenvs,ABleftenvs,ABrightenvs,scatter);
end

function MPSKit.leftenv(env::ScatterOverlap,row::Int,col::Int,st::LeftGaugedMW)
    a = findfirst(i -> !(st.AL[row,i] === env.ldependencies[row,i]), 1:(col-1))

    ham = env.opp;
    scatter = env.above;

    gs = scatter.B2.left_gs;

    K1 = scatter.B1.momentum;
    K2 = scatter.B2.momentum;

    fuser = isomorphism(utilleg(st),utilleg(scatter.B2)*utilleg(scatter.B1));

    if !isnothing(a)
        #we need to recalculate
        for j = a:col-1
            #leftenv
            env.leftenvs[row,j+1] = env.leftenvs[row,j] * TransferMatrix(gs.AL[row+j],ham[row+j],st.AL[row,j])

            #Aleftenv
            env.Aleftenvs[row,j+1] = env.Aleftenvs[row,j] * TransferMatrix(gs.AR[row+j],ham[row+j],st.AL[row,j])
            for (k,l) in keys(ham[row+j])
                @tensor  env.Aleftenvs[row,j+1][l][-1 -2;-3 -4] +=env.leftenvs[row,j][k][1 2;6 4]*conj(st.AL[row,j][1 5; -1])*fuser[6;-3 3]*ham[row+j][k,l][2 5;7 -2]*scatter.B1[row+j][4 7;3 -4]*exp(1im*K1*j)
            end

            #ABleftenv
            env.ABleftenvs[row,j+1] = env.ABleftenvs[row,j]*TransferMatrix(gs.AR[row+j],ham[row+j],st.AL[row,j])
            for (k,l) in keys(ham[row+j-1])
                @tensor  env.ABleftenvs[row,j+1][l][-1 -2;-3]+=  (env.Aleftenvs[row,j][k]*exp(1im*K2*j))[3 6;5 2]*inv(gs.CR[row+j-1])[2;4]*conj(st.AL[row,j][3 7;-1])*ham[row+j][k,l][6 7;1 -2]*scatter.B2[row+j][4 1;5 -3]
            end

            env.ldependencies[row,j] = st.AL[row,j]
        end
    end

    return (env.leftenvs[row,col],env.Aleftenvs[row,col],env.ABleftenvs[row,col]);
end

function MPSKit.rightenv(env::ScatterOverlap,row::Int,col::Int,st::LeftGaugedMW)
    a = findfirst(i -> !(st.AR[row,i] === env.rdependencies[row,i]), size(st,2):-1:(col+1))

    ham = env.opp;
    scatter = env.above;

    gs = scatter.B2.left_gs;

    K1 = scatter.B1.momentum;
    K2 = scatter.B2.momentum;

    fuser = isomorphism(utilleg(st),utilleg(scatter.B2)*utilleg(scatter.B1));

    if !isnothing(a)
        a = size(st,2)-a+1;
        #we need to recalculate
        for j = a:-1:col+1
            env.rightenvs[row,j] = TransferMatrix(gs.AR[row+j],ham[row+j],st.AR[row,j]) * env.rightenvs[row,j+1]

            env.Brightenvs[row,j] = TransferMatrix(gs.AL[row+j],ham[row+j],st.AR[row,j]) * env.Brightenvs[row,j+1]
            env.Brightenvs[row,j] += TransferMatrix(scatter.B2[row+j],ham[row+j],st.AR[row,j]) * env.rightenvs[row,j+1]*exp(1im*K2*j)

            env.ABrightenvs[row,j] = TransferMatrix(gs.AL[row+j],ham[row+j],st.AR[row,j]) * env.ABrightenvs[row,j+1]
                
            for (k,l) in keys(ham[j])
                @plansor  env.ABrightenvs[row,j][k][-1 -2;-3 -4] += scatter.B1[row+j][-1 2;4 1]*(fuser*exp(1im*K1*j))[-3;8 4]*inv(gs.CR[row+j])[1;3]*env.Brightenvs[row,j+1][l][3 5;8 7]*conj(st.AR[row,j][-4 6;7])*ham[row+j][k,l][-2 6;2 5]
            end

            env.rdependencies[row,j] = st.AR[row,j]
        end
    end

    return (env.rightenvs[row,col+1],env.Brightenvs[row,col+1],env.ABrightenvs[row,col+1]);
end

function MPSKit.ac_proj(row::Int,col::Int,st::LeftGaugedMW,env::ScatterOverlap)
    ham = env.opp;
    E = env.E;
    scatter = env.above;

    gs = scatter.B2.left_gs;

    K1 = scatter.B1.momentum;
    K2 = scatter.B2.momentum;

    (left,aleft,ableft) = leftenv(env,row,col,st);
    (right,bright,abright) = rightenv(env,row,col,st);
    
    fuser = isomorphism(utilleg(st),utilleg(scatter.B2)*utilleg(scatter.B1));

    #local total
    total = zero(st.AC[row,col]);
    pos = row+col;
    
    for (j,k) in keys(ham[pos])
        
        @tensor t[-1 -2;-3] := (aleft[j]*exp(1im*K2*col))[-1,7,5,6]*inv(gs.CR[pos-1])[6,1]*scatter.B2[pos][1,9,5,4]*right[k][4,8,-3]*ham[pos][j,k][7,-2,9,8];
        @tensor t[-1 -2;-3] += (left[j]*exp(1im*K1*col))[-1,6,8,7]*fuser[8,3,9]*scatter.B1[pos][7,5,9,1]*inv(gs.CR[pos])[1,2]*bright[k][2,4,3,-3]*ham[pos][j,k][6,-2,5,4];

        @tensor t[-1 -2;-3] += aleft[j][-1,5,7,6]*gs.AR[pos][6,2,1]*inv(gs.CR[pos])[1,3]*bright[k][3,4,7,-3]*ham[pos][j,k][5,-2,2,4]

        
        
        @tensor t[-1 -2;-3] += ableft[j][-1,2,3]*gs.AR[pos][3,1,4]*right[k][4,5,-3]*ham[pos][j,k][2,-2,1,5];
        @tensor t[-1 -2;-3] += left[j][-1,5,4,2]*gs.AL[pos][2,6,3]*abright[k][3,1,4,-3]*ham[pos][j,k][5,-2,6,1];
        
        total += t;
    end
    
    @tensor total[-1 -2;-3] -= E*(exp(1im*K2*col)*aleft[1])[-1 7;5 6]*inv(gs.CR[pos-1])[6,1]*scatter.B2[pos][1,-2,5,4]*right[end][4,7,-3];
    @tensor total[-1 -2;-3] -= (E*exp(1im*K1*col)*left[1])[-1,4,8,7]*fuser[8,3,9]*scatter.B1[pos][7,-2,9,1]*inv(gs.CR[pos])[1,2]*bright[end][2,4,3,-3];
    @tensor total[-1 -2;-3] -= (E*ableft[1])[-1,2,3]*gs.AR[pos][3,-2,1]*right[end][1,2,-3];
    @tensor total[-1 -2;-3] -= (E*left[1])[-1,7,8,3]*gs.AL[pos][3,-2,6]*abright[end][6,7,8,-3];
    @tensor total[-1 -2;-3] -= (E*aleft[1])[-1,4,7,6]*gs.AR[pos][6,-2,1]*inv(gs.CR[pos])[1,3]*bright[end][3,4,7,-3]
    
    total
end

function MPSKit.approximate!(init::LeftGaugedMW,sq::Tuple,alg,envs=environments(init,sq))
    tor =  approximate!(init,[sq],alg,[envs]);
    return (tor[1],tor[2][1],tor[3])
end
function MPSKit.approximate!(init::LeftGaugedMW, squash::Vector,alg::DMRG,envs = [environments(init,sq) for sq in squash])

    tol=alg.tol;maxiter=alg.maxiter
    iter = 0; delta = 2*tol

    while iter < maxiter && delta > tol
        delta=0.0

        #finalize
        (init,envs) = alg.finalize(iter,init,squash,envs);

        for row in 1:size(init,1),col in [1:size(init,2);(size(init,2)-1):-1:2]
            newac = sum(map(zip(squash,envs)) do (sq,pr)
                MPSKit.ac_proj(row,col,init,pr)
            end)

            delta = max(delta,norm(newac-init.AC[row,col])/norm(newac))

            init.AC[row,col] = newac
        end

        alg.verbose && @info "dmrg iter $(iter) error $(delta)"
        flush(stdout);

        iter += 1
    end

    delta > tol && @warn "dmrg failed to converge $(delta)>$(tol)"
    return init,envs,delta
end


#%%
function left_double_env(K,B2,lB1,ham,fuse_fun)
    inv_C = inv.(B2.left_gs.CR);
    @assert isempty(filter(x->MPSKit.isid(ham,x),2:ham.odim-1));

    O_type = eltype(ham[1]);
    lBs = PeriodicArray{O_type,2}(undef,ham.odim,length(B2));

    for pos in 1:length(B2)
        # instantiate
        for i in 1:ham.odim
            lBs[i,pos+1] = TensorMap(zeros,eltype(lB1[1,pos+1]),_firstspace(lB1[i,pos+1])*space(lB1[i,pos+1],2),_firstspace(fuse_fun)'*_lastspace(lB1[i,pos+1])');
        end

        for (i,j) in keys(ham[pos])
            @tensor lBs[j,pos+1][-1 -2;-3 -4] += lB1[i,pos][7 6;3 1]*inv_C[pos-1][1;2]*B2[pos][2 5;4 -4]*conj(B2.left_gs.AL[pos][7 8;-1])*ham[pos][i,j][6 8;5 -2]*fuse_fun[-3;4 3]
        end
    end

    lBs.*=exp(-1im*K);
    for i in 2:length(B2)
        lBs[:,i+1] += (lBs[:,i]*TransferMatrix(B2.right_gs.AR[i],ham[i],B2.right_gs.AL[i]))*exp(-1im*K);
    end

    for pos in 1:length(B2),i in (1,ham.odim)
        @plansor lBs[i,pos+1][-1 -2;-3 -4] -= lBs[i,pos+1][1 4;-3 2]*r_RL(B2.left_gs,pos)[2;3]*τ[3 4;5 1]*l_RL(B2.left_gs,pos+1)[-1;6]*τ[5 6;-4 -2]
    end


    lBE = MPSKit.left_excitation_transfer_system(lBs[:,1],ham,B2;mom = K)

    lBs[:,1] = lBE;
    for i=1:length(B2)-1
        lBE = (lBE*TransferMatrix(B2.right_gs.AR[i],ham[i],B2.right_gs.AL[i]))/exp(1im*K);

        lBs[:,i+1] += lBE;
    end

    lBs
end

function right_double_env(K,B2,rB1,ham,fuse_fun)
    inv_C = inv.(B2.right_gs.CR);
    @assert isempty(filter(x->MPSKit.isid(ham,x),2:ham.odim-1));

    O_type = eltype(ham[1]);
    rBs = PeriodicArray{O_type,2}(undef,ham.odim,length(B2));
    
    for pos in 1:length(inv_C) 
        for i in 1:ham.odim
            rBs[i,pos-1] = TensorMap(zeros,eltype(rB1[1,pos-1]),_firstspace(rB1[i,pos-1])*space(rB1[i,pos-1],2),_firstspace(fuse_fun)'*_lastspace(rB1[i,pos-1])')
        end


        for (i,j) in keys(ham[pos])
            @tensor rBs[i,pos-1][-1 -2;-3 -4] += B2[pos][-1 5;3 1]*inv_C[pos][1;2]*rB1[j,pos][2  6;4 8]*fuse_fun[-3;4 3]*ham[pos][i,j][-2 7;5 6]*conj(B2.left_gs.AR[pos][-4 7;8])
        end
    end
    
    rBs.*=exp(1im*K);
    
    for i in length(inv_C)-1:-1:1
        rBs[:,i-1] += TransferMatrix(B2.left_gs.AL[i],ham[i],B2.left_gs.AR[i])*(rBs[:,i]*exp(1im*K))
    end

    for pos in 1:length(inv_C),i in (1,ham.odim)
        @plansor rBs[i,pos-1][-1 -2;-3 -4] -= τ[6 4;1 3]*rBs[i,pos-1][1 3;-3 2]*l_LR(B2.left_gs,pos)[2;4]*r_LR(B2.left_gs,pos-1)[-1;5]*τ[-2 -4;5 6]
    end
    

    rBE = MPSKit.right_excitation_transfer_system(rBs[:,end],ham,B2;mom = K)

    rBs[:,end] = rBE;

    for i=length(B2):-1:2
        rBE = TransferMatrix(B2.left_gs.AL[i],ham[i],B2.right_gs.AR[i])*rBE*exp(1im*K);
        rBs[:,i-1] += rBE
    end

    rBs
end
