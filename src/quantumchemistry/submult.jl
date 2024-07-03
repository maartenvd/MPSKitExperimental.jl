
struct LeftSubMult{I}
    table::Dict{Tuple{I,I},Vector{Tuple{UnitRange{Int},UnitRange{Int},UnitRange{Int},UnitRange{Int}}}}
end

function LeftSubMult(homsp1,homsp2)
    # determine output domspace
    outspaces = [sp for sp in domain(homsp2)]
    for l in length(domain(homsp2))+1:length(domain(homsp1))
        push!(outspaces,domain(homsp1)[l])
    end
    outspace = reduce(*,outspaces)

    Nsubmult = length(codomain(homsp2))
    
    @assert Nsubmult > 1 # probably doesn't work?

    # iterate over domain(homsp1)
    # iterate over outspace

    blocksectoriterator_A = blocksectors(homsp1); # blocks in tensor A
    colr_A, _ = TensorKit._buildblockstructure(domain(homsp1), blocksectoriterator_A)

    blocksectoriterator_B = blocksectors(homsp2);
    rowr_B, _ = TensorKit._buildblockstructure(codomain(homsp2), blocksectoriterator_B)
    colr_B, _ = TensorKit._buildblockstructure(domain(homsp2), blocksectoriterator_B)

    blocksectoriterator_dst = blocksectors(outspace)
    colr_dst, _ = TensorKit._buildblockstructure(outspace, blocksectoriterator_dst)

    table = Dict{Tuple{sectortype(homsp1),sectortype(homsp1)},Vector{Tuple{UnitRange{Int},UnitRange{Int},UnitRange{Int},UnitRange{Int}}}}()
    for block in blocksectors(homsp1)
        
        # block ∈ A
        # get all fusiontrees in tensor A that fuse to this block
        for (f_A, c_A) in colr_A[block]
            coupled_B = f_A.innerlines[Nsubmult-1]
            coupled_B in keys(rowr_B) || continue


            # find the correct fusiontree in cur_rowr_B for the given f_A
            hit_r_B = 0:-1
            for (f_B,r_B) in rowr_B[coupled_B]
                f_B.uncoupled == f_A.uncoupled[1:length(f_B.uncoupled)] || continue
                f_B.innerlines == f_A.innerlines[1:length(f_B.innerlines)] || continue
                @assert hit_r_B == 0:-1
                hit_r_B = r_B
            end

          
            # for every possible cur_colr_B + f_A, find the correct cur_colr_dst

            for (f_B,c_B) in colr_B[coupled_B], (f_dst,c_dst) in colr_dst[f_A.coupled]
                f_B.uncoupled == f_dst.uncoupled[1:length(f_B.uncoupled)] || continue
                f_B.innerlines == f_dst.innerlines[1:length(f_B.innerlines)] || continue
                


                f_dst.uncoupled[length(f_B.uncoupled)+1:end] == f_A.uncoupled[Nsubmult+1:end] || continue
                f_dst.innerlines[length(f_B.innerlines)+2:end] == f_A.innerlines[Nsubmult:end] || continue 

                f_dst.innerlines[length(f_B.innerlines)+1] == coupled_B || continue
                
                if !haskey(table,(block,coupled_B))
                    table[(block,coupled_B)] = []
                end
                push!(table[(block,coupled_B)],(c_A,hit_r_B,c_B,c_dst))
            end
        end
    end

    return LeftSubMult(table)
end


function (fct::LeftSubMult)(t_dst,t_A,t_B)
    for ((sect_A,sect_B),sub_blocks) in fct.table
        dst_block = t_dst.data[sect_A]
        A_block = t_A.data[sect_A]
        B_block = t_B.data[sect_B]
        for (col_A,row_B,col_B,col_dst) in sub_blocks
            #dst_block[:,col_dst] += (@view A_block[:,col_A])*(@view B_block[row_B,col_B])
            mul!((@view dst_block[:,col_dst]),(@view A_block[:,col_A]),(@view B_block[row_B,col_B]),true,true)
        end
    end
end
