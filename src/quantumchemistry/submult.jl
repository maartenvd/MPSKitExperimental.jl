struct LeftSubMult
    table::Vector{Tuple{Tuple{UnitRange{Int},Int,Int},Tuple{UnitRange{Int},Int,Int},Tuple{UnitRange{Int},Int,Int},Tuple{UnitRange{Int},UnitRange{Int},UnitRange{Int},UnitRange{Int}}}}
    #Dict{Tuple{I,I},Vector{Tuple{UnitRange{Int},UnitRange{Int},UnitRange{Int},UnitRange{Int}}}}
end

function rowr_colr_from_fusionblockstructure(structure::TensorKit.FusionBlockStructure{I,F₁,F₂}) where {I,F₁,F₂}
    rowr = Dict{F₁,UnitRange{Int}}()
    colr = Dict{F₂,UnitRange{Int}}()
    
    for ((f1,f2),(sz,st,o)) in zip(structure.fusiontreelist,structure.fusiontreestructure)
        (block_sz,block_range) = structure.blockstructure[f1.coupled]
        block_range_start = block_range[1]
        
        @assert st[1] == 1

        i = mod(o+1-block_range_start,block_sz[1])+1
        j = (o+1-block_range_start)÷block_sz[1]+1

        irange = i:i+sz[1]-1
        jrange = j:j+sz[2]-1

        if f1 in keys(rowr)
            @assert rowr[f1] == irange
        else
            rowr[f1] = irange
        end

        if f2 in keys(colr)
            @assert colr[f2] == jrange
        else
            colr[f2] = jrange
        end
    end

    return (rowr,colr)
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

    block_A = TensorKit.fusionblockstructure(homsp1)
    block_B = TensorKit.fusionblockstructure(homsp2)
    block_dst = TensorKit.fusionblockstructure(codomain(homsp1)←outspace)

    table = Dict{Tuple{sectortype(homsp1),sectortype(homsp1)},Vector{Tuple{UnitRange{Int},UnitRange{Int},UnitRange{Int},UnitRange{Int}}}}()

    (rowr_A,colr_A) = rowr_colr_from_fusionblockstructure(block_A)
    (rowr_B,colr_B) = rowr_colr_from_fusionblockstructure(block_B)
    (rowr_dst,colr_dst) = rowr_colr_from_fusionblockstructure(block_dst)

    for (f_A,c_A) in colr_A
        block = f_A.coupled

                
        # block ∈ A
        # get all fusiontrees in tensor A that fuse to this block

        coupled_B = f_A.innerlines[Nsubmult-1]
        coupled_B in keys(block_B.blockstructure) || continue


        # find the correct fusiontree in cur_rowr_B for the given f_A
        # this for loop can be replaced by a direct dict lookup
        hit_r_B = 0:-1
        for (f_B,r_B) in rowr_B
            f_B.coupled == coupled_B || continue
            f_B.uncoupled == f_A.uncoupled[1:length(f_B.uncoupled)] || continue
            f_B.innerlines == f_A.innerlines[1:length(f_B.innerlines)] || continue
            @assert hit_r_B == 0:-1
            hit_r_B = r_B
        end

        
        # for every possible cur_colr_B + f_A, find the correct cur_colr_dst
        for (f_B,c_B) in colr_B, (f_dst,c_dst) in colr_dst
            f_B.coupled == coupled_B && f_dst.coupled == block || continue

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
    
    # try to simplify - this is not the optimal way to do this
    # but on the other hand, it's compatible with the old version of tensorkit and the potential new one
    for ((sect_A,sect_B),sub_blocks) in table
        # if sub_blocks describe neighboring regions, then we can merge
        # (r1,r2,r3,r4) = sub_blocks
        # dst[:,r4] = A[:,r1]*B[r2,r3]
        # if two blocks have the same r1,r2 but successive r3/r4 then they can be merged
        # if two blocks have the same r3,r4 but successive r1,r2 then they can be merged
        
        d = Dict{Tuple{UnitRange{Int},UnitRange{Int}},Vector{Tuple{UnitRange{Int},UnitRange{Int}}}}()
        for (r1,r2,r3,r4) in sub_blocks
            if (r1,r2) in keys(d)
                # check if any can be merged
                hit_1 = findfirst(x-> x[1].start-1 == r3.stop && x[2].start-1 == r4.stop,d[(r1,r2)])
                hit_2 = findfirst(x-> x[1].stop+1 == r3.start && x[2].stop+1 == r4.start,d[(r1,r2)])

                if isnothing(hit_1) && isnothing(hit_2)
                    push!(d[(r1,r2)],(r3,r4))
                elseif isnothing(hit_1)
                    (a1,a2) = d[(r1,r2)][hit_2]
                    d[(r1,r2)][hit_2] = (a1.start:r3.stop,a2.start:r4.stop)
                elseif isnothing(hit_2)
                    (a1,a2) = d[(r1,r2)][hit_1]
                    d[(r1,r2)][hit_1] = (r3.start:a1.stop,r4.start:a2.stop)
                else
                    (a1,a2) = d[(r1,r2)][hit_1]
                    (b1,b2) = d[(r1,r2)][hit_2]
                    d[(r1,r2)][hit_1] = (b1.start:a1.stop,b2.start:a2.stop)
                    deleteat!(d[(r1,r2)],hit_2)
                end
                
            else
                d[(r1,r2)] = [(r3,r4)]
            end
        end

        sub_blocks = reduce(vcat,[[(r1,r2,r3,r4) for (r3,r4) in d[(r1,r2)]] for (r1,r2) in keys(d)])
        empty!(d)

        for (r1,r2,r3,r4) in sub_blocks
            if (r3,r4) in keys(d)
                # check if any can be merged
                hit_1 = findfirst(x-> x[1].start-1 == r1.stop && x[2].start-1 == r2.stop,d[(r3,r4)])
                hit_2 = findfirst(x-> x[1].stop+1 == r1.start && x[2].stop+1 == r2.start,d[(r3,r4)])

                if isnothing(hit_1) && isnothing(hit_2)
                    push!(d[(r3,r4)],(r1,r2))
                elseif isnothing(hit_1)
                    (a1,a2) = d[(r3,r4)][hit_2]
                    d[(r3,r4)][hit_2] = (a1.start:r1.stop,a2.start:r2.stop)
                elseif isnothing(hit_2)
                    (a1,a2) = d[(r3,r4)][hit_1]
                    d[(r3,r4)][hit_1] = (r1.start:a1.stop,r2.start:a2.stop)
                else
                    (a1,a2) = d[(r3,r4)][hit_1]
                    (b1,b2) = d[(r3,r4)][hit_2]
                    d[(r3,r4)][hit_1] = (b1.start:a1.stop,b2.start:a2.stop)
                    deleteat!(d[(r3,r4)],hit_2)
                end
                
            else
                d[(r3,r4)] = [(r1,r2)]
            end
        end

        table[(sect_A,sect_B)]= reduce(vcat,[[(r1,r2,r3,r4) for (r1,r2) in d[(r3,r4)]] for (r3,r4) in keys(d)])

    end
    
    finaltable = Vector{Tuple{Tuple{UnitRange{Int},Int,Int},Tuple{UnitRange{Int},Int,Int},Tuple{UnitRange{Int},Int,Int},Tuple{UnitRange{Int},UnitRange{Int},UnitRange{Int},UnitRange{Int}}}}()

    for ((sect_A,sect_B),vs) in table
        ((d1,d2),dr) = block_dst.blockstructure[sect_A]
        ((a1,a2),ar) = block_A.blockstructure[sect_A]
        ((b1,b2),br) = block_B.blockstructure[sect_B]

        for v in vs
            push!(finaltable,((dr,d1,d2),(ar,a1,a2),(br,b1,b2),v))
        end
    end

    return LeftSubMult(finaltable)
end


using StridedViews
function (fct::LeftSubMult)(t_dst,t_A,t_B)
    @inbounds for (sect_dst,sect_A,sect_B,(col_A,row_B,col_B,col_dst)) in fct.table
        dst_block = StridedView(t_dst.data, (sect_dst[2],sect_dst[3]), (1, sect_dst[2]), sect_dst[1].start-1) #reshape(view(t_dst.data,sect_dst[1]),sect_dst[2],sect_dst[3])
        A_block = StridedView(t_A.data, (sect_A[2],sect_A[3]), (1, sect_A[2]), sect_A[1].start-1)#reshape(view(t_A.data,sect_A[1]),sect_A[2],sect_A[3])
        B_block = StridedView(t_B.data, (sect_B[2],sect_B[3]), (1, sect_B[2]), sect_B[1].start-1)#reshape(view(t_B.data,sect_B[1]),sect_B[2],sect_B[3])
        #dst_block = block(t_dst,sect_A)
        #A_block = block(t_A,sect_A)
        #B_block = block(t_B,sect_B)
   
        #dst_block[:,col_dst] += (@view A_block[:,col_A])*(@view B_block[row_B,col_B])
        mul!(( dst_block[:,col_dst]),( A_block[:,col_A]),( B_block[row_B,col_B]),true,true)
    end
end
