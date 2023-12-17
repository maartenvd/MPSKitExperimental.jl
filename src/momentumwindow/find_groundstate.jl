function MPSKit.find_groundstate(st::LeftGaugedMW,h::MPOHamiltonian,alg::DMRG,envs = environments(st,h))
    val = 0.0

    for topit in 1:alg.maxiter
        err = 0
        for i in 1:size(st,2)
            orig = RecursiveVec(st.AC[:,i])
            (vals,vecs,convhist) = eigsolve(orig,1,:SR,alg.eigalg) do x
                st.AC[:,i] = x.vecs[:];
                envs = environments(st,h,envs.le,envs.re);
                res = map(1:size(st,1)) do row
                    MPSKit.ac_proj(row,i,st,envs)
                end
                RecursiveVec(res)
            end
            val = vals[1]
            v = vecs[1]

            
            err = max(1-abs(dot(v,orig))^2,err)

            st.AC[:,i] = v.vecs[:];
        end

        @show topit,err
        if err < alg.tol
            break
        end
    end

    return st,val
end