#%% we also need the effective excitation hamiltonian for the leftgauged momentumwindow
struct MWenv{A<:MPSTensor,B<:MPOTensor,C<:LeftGaugedMW,O<:MPOHamiltonian,D} <: Cache
    lefties::PeriodicArray{Vector{A},3}
    righties::PeriodicArray{Vector{A},3}

    left_above::PeriodicArray{Vector{B},2}
    left_below::PeriodicArray{Vector{B},2}

    right_above::PeriodicArray{Vector{A},2}
    right_below::PeriodicArray{Vector{A},2}

    lBEs::PeriodicArray{Vector{A},2}
    rBEs::PeriodicArray{Vector{B},2}

    above::C

    left_dependencies::Matrix{A}
    right_dependencies::Matrix{A}

    opp::O

    le::D
    re::D
end

MPSKit.environments(st::LeftGaugedMW,ham::MPOHamiltonian,le = environments(st.left_gs,ham),re = environments(st.right_gs,ham);kwargs...) = environments(st,(ham,st),le,re;kwargs...)
function MPSKit.environments(below::LeftGaugedMW,toapprox::Tuple{<:MPOHamiltonian,<:LeftGaugedMW},le = environments(below.left_gs,first(toapprox)),re = environments(below.right_gs,first(toapprox)))
    # - support for different above/below
    (ham,above) = toapprox;

    (above.left_gs === below.left_gs && above.right_gs === below.right_gs && utilleg(above) == utilleg(below) && above.momentum == below.momentum) || throw(ArgumentError("not supported (or sensical for that matter)"))

    K = above.momentum;

    #threeleg type
    treeleg_type = typeof(above.left_gs.AL[1,1]);
    fourleg_type = tensormaptype(spacetype(treeleg_type),2,2,storagetype(treeleg_type))

    #=
    first index == fysical position
    second index == the collumn
    =#
    left_above = PeriodicArray{Vector{fourleg_type},2}(undef,size(above,1),size(above,2)+1);
    left_below = PeriodicArray{Vector{fourleg_type},2}(undef,size(above,1),size(below,2)+1);
    right_above = PeriodicArray{Vector{treeleg_type},2}(undef,size(above,1),size(above,2)+1);
    right_below = PeriodicArray{Vector{treeleg_type},2}(undef,size(above,1),size(below,2)+1);


    for row in 1:size(above,1)
        #first element is special
        left_above[row+1,1] = (leftenv(le,row,above.left_gs)*exp(-1im*K))*TransferMatrix(above.VLs[row],ham[row],below.left_gs.AL[row]);
        left_below[row+1,1] = (leftenv(le,row,below.left_gs)*exp(1im*K))*TransferMatrix(above.left_gs.AL[row],ham[row],below.VLs[row]);
            

        right_above[row,end] = rightenv(re,row,above.right_gs);
        right_below[row,end] = rightenv(re,row,below.right_gs);
    end

    #the rest is just a matter of transferring
    for col in 1:size(above,2),fyspos in 1:size(above,1)
        left_above[fyspos+1,col+1] = MPSKit.transfer_left(left_above[fyspos,col],ham[fyspos],above.AL[fyspos-col,col],above.left_gs.AL[fyspos])*exp(-1im*K);

        ncol = size(above,2)-col+1;
        right_above[fyspos-1,ncol] = transfer_right(right_above[fyspos,ncol+1],ham[fyspos],above.AR[fyspos-ncol,ncol],above.right_gs.AR[fyspos]);
    end

    #=
    doe die grote matrix
    first index - physical position
    second index - column above
    third index - column below
    =#
    lefties = PeriodicArray{Vector{treeleg_type},3}(undef,size(above,1),size(above,2)+1,size(below,2)+1);
    righties = PeriodicArray{Vector{treeleg_type},3}(undef,size(above,1),size(above,2)+1,size(below,2)+1);

    #fill it in
    for row in 1:size(above,1)
        lefties[row+1,1,1] = leftenv(le,row,above.left_gs) * FusingTransferMatrix(above.VLs[row],ham[row],below.VLs[row])
            

        for col in 1:size(above,2)
            lefties[row+col+1,col+1,1] = left_above[row+col,col] * FusingTransferMatrix(above.AL[row,col],ham[row+col],below.VLs[row+col])
        end
    end

    for row in 1:size(above,1),
        col in 1:size(above,2)+1
        righties[row+col,col,end] = right_above[row+col,col];
    end

    lBs = map(enumerate(left_above[:,end])) do (i,v)
        map(v) do s
            @tensor tv[-1 -2;-3 -4] := s[-1,-2,-3,1] * above.CR[i-size(above,2)-1,end][1,-4]
        end
    end
    lB = copy(lBs[mod1(2,end)]);
    for i in 2:size(above,1)
        lB = lB*TransferMatrix(above.right_gs.AR[i],ham[i],above.left_gs.AL[i])*exp(-1im*K);
        lB += lBs[mod1(i+1,end)]
    end
    rBs = map(enumerate(right_above[:,1])) do (i,v)
        t = map(v) do s
            @tensor tv[-1 -2;-3] := above.CR[i,0][-1,1]*s[1,-2,-3]
        end
        MPSKit.transfer_right(t,ham[i],above.VLs[i],below.right_gs.AR[i])*exp(1im*K);
    end
    rBs = circshift(rBs,-1);
    rB = copy(rBs[mod1(-1,end)])
    for i in size(above,1)-1:-1:1
        rB = MPSKit.transfer_right(rB,ham[i],above.left_gs.AL[i],above.right_gs.AR[i])*exp(1im*K);
        rB += rBs[mod1(i-1,end)]
    end

    for i in (1,ham.odim)
        @plansor lB[i][-1 -2;-3 -4] -= lB[i][1 4;-3 2]*r_RL(above.right_gs,0)[2;3]*τ[3 4;5 1]*l_RL(above.right_gs,1)[-1;6]*τ[5 6;-4 -2]
        @plansor rB[i][-1 -2;-3 -4] -= τ[6 4;1 3]*rB[i][1 3;-3 2]*l_LR(above.left_gs,1)[2;4]*r_LR(above.left_gs,0)[-1;5]*τ[-2 -4;5 6]
    end

    lBs[1] = MPSKit.left_excitation_transfer_system(lB,ham,above);
    rBs[end] = MPSKit.right_excitation_transfer_system(rB,ham,above);
    
    for row in 2:size(above,1)
        lBs[row] += MPSKit.transfer_left(lBs[row-1],ham[row-1],above.right_gs.AR[row-1],above.left_gs.AL[row-1])*exp(-1im*K);

        row = size(above,1)-row+1
        rBs[row] += MPSKit.transfer_right(rBs[row+1],ham[row+1],above.left_gs.AL[row+1],above.right_gs.AR[row+1])*exp(1im*K);
    end

    #first index = fysical position, second index is collumn
    lBEs = PeriodicArray{Vector{treeleg_type},2}(undef,size(above,1),size(below,2)+1);
    rBEs = PeriodicArray{Vector{fourleg_type},2}(undef,size(above,1),size(below,2)+1);

    for row in 1:size(above,1)
        lBEs[row+1,1] = lBs[row] * FusingTransferMatrix(above.right_gs.AR[row],ham[row],below.VLs[row]);
        
        lBEs[row+1,1] += map(lefties[row+1,end,1]) do v
            @tensor v[-1 -2;-3] := v[-1,-2,1]*above.CR[row-size(above,2),end][1,-3]
        end


        rBEs[row,end] = rBs[row]
    end

    deps = similar.(below.AL);

    MWenv(lefties,righties,left_above,left_below,right_above,right_below,lBEs,rBEs,above,copy(deps),copy(deps),ham,le,re);
end


function MPSKit.leftenv(env::MWenv,row::Int,col::Int,below)
    a = findfirst(i -> !(below.AL[row,i] === env.left_dependencies[row,i]), 1:(col-1))

    above = env.above;
    ham = env.opp;
    K = below.momentum;

    if !isnothing(a)
        #we need to recalculate
        for j = a:col-1

            fp = row+j

            #update all left_below
            env.left_below[fp+1,j+1] = env.left_below[fp,j]*TransferMatrix(below.left_gs.AL[fp],ham[fp],below.AL[row,j])*exp(1im*K);

            #update all lefties
            for k in 1:size(env.lefties,2)-1
                env.lefties[fp+1,k+1,j+1] = env.lefties[fp,k,j]* TransferMatrix(above.AL[fp-k,k],ham[fp],below.AL[row,j]);
            end
            env.lefties[fp+1,1,j+1] = env.left_below[fp,j]*FusingTransferMatrix(above.VLs[fp],ham[fp],below.AL[row,j]);

            #update all lBEs
            env.lBEs[fp+1,j+1] = env.lBEs[fp,j] * TransferMatrix(above.right_gs.AR[fp],ham[fp],below.AL[row,j])
            env.lBEs[fp+1,j+1] += map(env.lefties[fp+1,end,j+1]) do v
                @plansor v[-1 -2;-3] := v[-1 -2;1]*above.CR[fp-size(above,2),end][1;-3]
            end
            
            env.left_dependencies[row,j] = below.AL[row,j]
        end
    end

    return env.lefties[row+col,:,col],env.lBEs[row+col,col],env.left_below[row+col,col]
end

function MPSKit.rightenv(env::MWenv,row::Int,col::Int,below)
    a = findfirst(i -> !(below.AR[row,i] === env.right_dependencies[row,i]), size(below,2):-1:(col+1))
    

    above = env.above;
    ham = env.opp;
    K = below.momentum;

    if !isnothing(a)
        a = size(below,2)-a+1
        
        #we need to recalculate
        for j = a:-1:col+1
            fp = row+j

            env.right_below[fp-1,j] = TransferMatrix(below.right_gs.AR[fp],ham[fp],below.AR[row,j])*env.right_below[fp,j+1];

            for k in 1:size(env.righties,2)-1
                env.righties[fp-1,k,j]  = TransferMatrix(above.AR[fp-k,k],ham[fp],below.AR[row,j])*env.righties[fp,k+1,j+1]
            end
            env.righties[fp-1,end,j] = env.right_below[fp-1,j]

            #rBEs
            env.rBEs[fp-1,j] = TransferMatrix(above.left_gs.AL[fp],ham[fp],below.AR[row,j])*env.rBEs[fp,j+1]*exp(1im*K);
            @tensor t_AC[-1 -2;-3 -4] := above.VLs[fp][-1 -2;-3 1]*above.CR[fp,0][1;-4]
            env.rBEs[fp-1,j] += TransferMatrix(t_AC,ham[fp],below.AR[row,j])*env.righties[fp,1,j+1]*exp(1im*K);

            env.right_dependencies[row,j] = below.AR[row,j]
        end
    end

    return env.righties[row+col,:,col+1],env.rBEs[row+col,col+1],env.right_below[row+col,col+1]
end

function MPSKit.ac_proj(row::Int,col::Int,below::LeftGaugedMW,env::MWenv)
    ham = env.opp;

    (lefties,lBE,left_below) = leftenv(env,row,col,below);
    (righties,rBE,right_below) = rightenv(env,row,col,below);

    fyspos = row+col;

    toret = zero(below.AC[row,col])

    for (j,k) in keys(ham[fyspos])
        @tensor t[-1 -2;-3] := lBE[j][-1,6,7]*env.above.right_gs.AR[fyspos][7,2,3]*right_below[k][3,4,-3]*ham[fyspos][j,k][6,-2,2,4]
        @tensor t[-1 -2;-3] += left_below[j][-1 3;4 5]*env.above.VLs[fyspos][5 2;4 6]*env.above.CR[fyspos,0][6;1]*righties[1][k][1 7;-3]*ham[fyspos][j,k][3 -2;2 7]
        @tensor t[-1 -2;-3] += left_below[j][-1,2,6,3]*env.above.left_gs.AL[fyspos][3,1,4]*rBE[k][4,5,6,-3]*ham[fyspos][j,k][2,-2,1,5]
        for i in 1:size(env.above,2)
            @tensor t[-1 -2;-3] += lefties[i][j][-1,4,5]*env.above.AC[fyspos-i,i][5,1,2]*righties[i+1][k][2,3,-3]*ham[fyspos][j,k][4,-2,1,3]
        end

        toret+=t
    end

    expv = expectation_value(below.left_gs,ham,row:(row+max(size(below,2),size(env.above,2))),env.le);

    if col <= size(env.above,2)
        @tensor toret[-1 -2;-3] -=(expv*lefties[col][1])[-1,2,1]*env.above.AC[row,col][1,-2,3]*righties[col+1][end][3,2,-3];
        
    else
        @tensor toret[-1 -2;-3] -=(expv*lBE[1])[-1,2,3]*env.above.right_gs.AR[fyspos][3,-2,4]*right_below[end][4,2,-3]
    end

    

    return toret
end

function MPSKit.c_proj(row::Int,col::Int,below::LeftGaugedMW,env::MWenv)
    ham = env.opp;

    (lefties,lBE,left_below) = leftenv(env,row,col+1,below);
    (righties,rBE,right_below) = rightenv(env,row,col,below);

    toret = similar(below.CR[row,col])

    fyspos = row+col;


    expv = expectation_value(below.left_gs,ham,row:(row+max(size(below,2),size(env.above,2))),env.le);
    if col <= size(env.above,2)
        @tensor toret[-1;-2] := -(expv*lefties[col+1][1])[-1,2,1]*env.above.CR[row,col][1;3]*righties[col+1][end][3,2,-2];
    else
        @tensor toret[-1;-2] := -(expv*lBE[1])[-1,2,1]*right_below[end][1,2,-2]
    end

    for j in 1:ham.odim
        for i in 0:size(env.above,2)-1
            @tensor toret[-1;-2] += lefties[i+1][j][-1,2,1]*env.above.CR[fyspos-i,i][1;3]*righties[i+1][j][3,2,-2]
        end
        @tensor toret[-1;-2] += lBE[j][-1,1,2]*right_below[j][2,1,-2]
        @tensor toret[-1;-2] += left_below[j][-1,2,6,1]*rBE[j][1,2,6,-2]*exp(-1im*below.momentum)

    end


    return toret
end

function s_proj(below::LeftGaugedMW,env::MWenv)
    sum(map(1:size(below,1)) do row
        c = MPSKit.c_proj(row,0,below,env);
        @tensor y[-1;-2] := below.VLs[row][1,2,-1,3]*(c*adjoint(below.CR[row,0]))[3,4]*conj(below.VLs[row][1,2,-2,4])
    end)
end
