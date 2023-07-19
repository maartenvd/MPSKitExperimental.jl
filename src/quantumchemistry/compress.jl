function compress(ham::FusedMPOHamiltonian{E,O,Sp}) where {E,O,Sp}
    mapped = [Int[] for i in 1:length(ham)+1];

    len = length(ham);
    bl_elt = eltype(ham[1].blocks);
    n_lstartblocks::Vector{bl_elt} = map(ham[1].blocks) do (lmask,lblock,e,rblock,rmask)
        lmask = [lmask[1]];
        lblock = [lblock[1]];

        (lmask,lblock,e,rblock,rmask)::bl_elt
    end

    filter!(n_lstartblocks) do (lmask,lblock,e,rblock,rmask)
        lmask[1]
    end
    n_ldomspaces = [ham[1].domspaces[1]];
    ham.data[1] = FusedSparseBlock{E,O,Sp}(n_ldomspaces,ham[1].imspaces,ham[1].pspace,n_lstartblocks);
    mapped[1] = [1];

    n_rstartblocks::Vector{bl_elt} = map(ham[len].blocks) do (lmask,lblock,e,rblock,rmask)
        rmask = [rmask[end]];
        rblock = [rblock[end]];

        (lmask,lblock,e,rblock,rmask)::bl_elt
    end

    filter!(n_rstartblocks) do (lmask,lblock,e,rblock,rmask)
        rmask[end]
    end
    mapped[len] = [length(ham[len].imspaces)];
    n_rimspaces = [ham[len].imspaces[end]]
    ham.data[len] = FusedSparseBlock{E,O,Sp}(ham[len].domspaces,n_rimspaces,ham[len].pspace,n_rstartblocks);
    
    for i in 1:length(ham)-1
        mapped[i+1] = collect(1:length(ham[i].imspaces))
    end

    for i in [1:length(ham)-1;length(ham)-2:-1:1]
        block_1 = ham[i];
        block_2 = ham[i+1];

        red_im = reduce((a,b)->a.||b,map(last,block_1.blocks));
        red_dom = reduce((a,b)->a.||b,map(first,block_2.blocks));
        red = red_im.&&red_dom
        mapped[i+1] = mapped[i+1][red];

        nl_imspaces = block_1.imspaces[red];
        nl_blocks::Vector{bl_elt} = map(ham[i].blocks) do (lmask,lblock,e,rblock,rmask)
            
            (lmask,lblock,e,rblock[red[rmask]],rmask[red])
        end

        nr_domspaces = block_2.domspaces[red];
        nr_blocks::Vector{bl_elt} = map(ham[i+1].blocks) do (lmask,lblock,e,rblock,rmask)
            (lmask[red],lblock[red[lmask]],e,rblock,rmask)
        end
        filter!(nl_blocks)  do (lmask,lblock,e,rblock,rmask)
            sum(rmask)>0
        end
        filter!(nr_blocks)  do (lmask,lblock,e,rblock,rmask)
            sum(lmask)>0
        end

        ham.data[i] = FusedSparseBlock{E,O,Sp}(ham[i].domspaces,nl_imspaces,ham[i].pspace,nl_blocks);
        ham.data[i+1] = FusedSparseBlock{E,O,Sp}(nr_domspaces,ham[i+1].imspaces,ham[i+1].pspace,nr_blocks);
    end

    for i in 1:length(ham)
        #@show length(ham[i].blocks),length(ham[i].domspaces),length(ham[i].imspaces)
    end

    return ham,mapped
end