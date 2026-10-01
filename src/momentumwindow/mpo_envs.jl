struct MPOMWenv{A<:MPSTensor,B1,B2,C<:LeftGaugedMW,O<:MPOHamiltonian,D} 
    lefties::PeriodicArray{B1,3}
    righties::PeriodicArray{B1,3}

    left_above::PeriodicArray{B2,2}
    left_below::PeriodicArray{B2,2}

    right_above::PeriodicArray{B1,2}
    right_below::PeriodicArray{B1,2}

    lBEs::PeriodicArray{B1,2}
    rBEs::PeriodicArray{B2,2}

    above::C

    left_dependencies::Matrix{A}
    right_dependencies::Matrix{A}

    opp::O # either dense mpo or sparse mpo

    le::D
    re::D
end

MPSKit.environments(st::LeftGaugedMW,ham::MPSKit.MPOTensor,le = environments(st.left_gs,ham),re = environments(st.right_gs,ham);kwargs...) = environments(st,(ham,st),le,re;kwargs...)
function MPSKit.environments(below::LeftGaugedMW,toapprox::Tuple{<:MPSKit.MPOTensor,<:LeftGaugedMW},le = environments(below.left_gs,first(toapprox)),re = environments(below.right_gs,first(toapprox)))
    (ham,above) = toapprox;
    (above.left_gs === below.left_gs && above.right_gs === below.right_gs && auxiliaryspace(above) == auxiliaryspace(below) && above.momentum == below.momentum) || throw(ArgumentError("not supported (or sensical for that matter)"))

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
            @tensor tv[-1 -2;-3 -4] := s[-1,-2,-3,1] * above.C[i-size(above,2)-1,end][1,-4]
        end
    end
    lB = copy(lBs[mod1(2,end)]);
    for i in 2:size(above,1)
        lB = lB*TransferMatrix(above.right_gs.AR[i],ham[i],above.left_gs.AL[i])*exp(-1im*K);
        #MPSKit.transfer_left(lB,ham[i],above.right_gs.AR[i],above.left_gs.AL[i])*exp(-1im*K);
        lB += lBs[mod1(i+1,end)]
    end

    lBs[1] = MPSKit.left_excitation_transfer_system(lB,ham,above);

    rBs = map(enumerate(right_above[:,1])) do (i,v)
        t = map(v) do s
            @tensor tv[-1 -2;-3] := above.C[i,0][-1,1]*s[1,-2,-3]
        end

        MPSKit.transfer_right(t,ham[i],above.VLs[i],below.right_gs.AR[i])*exp(1im*K);
    end
    rBs = circshift(rBs,-1);
    rB = copy(rBs[mod1(-1,end)])
    for i in size(above,1)-1:-1:1
        rB = MPSKit.transfer_right(rB,ham[i],above.left_gs.AL[i],above.right_gs.AR[i])*exp(1im*K);
        rB += rBs[mod1(i-1,end)]
    end

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
            @tensor v[-1 -2;-3] := v[-1,-2,1]*above.C[row-size(above,2),end][1,-3]
        end


        rBEs[row,end] = rBs[row]
    end

    deps = similar.(below.AL);

    MPOMWenv(lefties,righties,left_above,left_below,right_above,right_below,lBEs,rBEs,above,copy(deps),copy(deps),ham,le,re);
end
