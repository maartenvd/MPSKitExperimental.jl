function orb_opt(ham::CASSCF_Ham,dV,dK;tol=MPSKit.Defaults.tol, maxiter = MPSKit.Defaults.maxiter)
    #=
    (E,dU) = orb_cfun((ham,dV,dK));
    for it in 1:maxiter
        update = orb_precondition((ham,dV,dK),dU);
        ((ham,dV,dK),_) = orb_retract((ham,dV,dK),update,1);
        (E,dU) = orb_cfun((ham,dV,dK))
        @info "gd $(E) $(norm(dU))"
        norm(dU) < tol && break;
    end

    return (ham,E)
    =#

    #return optimtest(orb_cfun,orbmanpoint(ham,dV,dK);alpha=-0.1:0.01:0.1,retract=orb_retract,inner=orb_inner)

    (out,fx,_,_,normgradhistory) = optimize(orb_cfun,orbmanpoint(ham,dV,dK), ConjugateGradient(gradtol=tol, maxiter=100, verbosity = 2);
    retract = orb_retract, inner = orb_inner, transport! = orb_transport! ,
    scale! = orb_scale! , add! = orb_add! , isometrictransport = true, precondition = orb_precondition);



    (nham,_) = out
    return (nham,fx)

end



function orbmanpoint(ham,dV,dK)
    ABs = get_A_B(ham);
    (ham,dV,dK,ABs,project_U(dV,dK,ham,ABs))
end


function project_U(odV,odK,ham::CASSCF_Ham,ABs = get_A_B(ham))
    #=
    energy should be V * odV + K * odK + E 
    odV
    =#

    tot = 1:ham.active.stop
    dV = zeros(eltype(odV),ham.active.stop,ham.active.stop,ham.active.stop,ham.active.stop);
    dK = zeros(eltype(odV),ham.active.stop,ham.active.stop);
    dV[ham.active,ham.active,ham.active,ham.active] .= odV;
    dK[ham.active,ham.active] .= odK;
    closed = 1:(first(ham.active)-1);


    for c in closed
        dK[c,c] = 2;

        dV[c,c,c,c] += 2
        for n in closed
            if n != c
                dV[n,c,c,n] += 4;
                #dV[c,n,n,c] += 4;
                dV[c,n,c,n] -= 2;
                #dV[n,c,n,c] -= 2;
            end
        end

        for n in ham.active,m in ham.active
            dV[n,c,c,m] += dK[n,m]*2;
            dV[c,n,m,c] += dK[n,m]*2;
            dV[c,n,c,m] -= dK[n,m];
            dV[n,c,m,c] -= dK[n,m];
        end
    end


    g = zero(ham.ao2mo);

    B = ham.ao2mo
    
    K = B'*(nuclear(ham.basis)+kinetic(ham.basis))*B;
    K_a_o = K[tot,:];
    @tensor g[tot,:][-1;-2] += -K_a_o[1;-2]*dK[1;-1];
    K_o_a = K[:,tot];
    @tensor g[:,tot][-1;-2] += K_o_a[-1;1]*dK[-2;1];

    (two_inds,As,Bs) = ABs

    for (ind,(ai,bi)) in enumerate(two_inds)
        tohandle = [(As[ind],Bs[ind],(ai,bi))];
        if ai!=bi
            push!(tohandle,(As[ind],Bs[ind]',(bi,ai)));
        end

        for (cur_A,cur_B,(cur_I,cur_J)) in tohandle
            temp_V = @view dV[:,cur_I,cur_J,:];
            temp_A = @view cur_A[:,tot];
            @tensor g[:,tot][-1,-2] += temp_A[-1,1]*temp_V[-2,1]

            temp_V = @view dV[cur_I,:,:,cur_J];
            temp_A = @view cur_A[:,tot];
            @tensor g[:,tot][-1,-2] += temp_A[-1,1]*temp_V[-2,1]


            temp_V = @view dV[:,cur_I,cur_J,:];
            temp_A = @view cur_A[tot,:];
            @tensor g[tot,:][-1,-2] -= temp_A[1,-2]*temp_V[1,-1]

            temp_V = @view dV[cur_I,:,:,cur_J];
            temp_A = @view cur_A[tot,:];
            @tensor g[tot,:][-1,-2] -= temp_A[1,-2]*temp_V[1,-1]
            
        end
    end




    function apply_hessian(x)
        hess_g = zero(x);


        #K = B'*(nuclear(h.basis)+kinetic(h.basis))*B;
        hess_K = x*K + K*x'
        K_a_o = hess_K[tot,:];
        @tensor hess_g[tot,:][-1;-2] += -K_a_o[1;-2]*dK[1;-1];
        
        K_o_a = hess_K[:,tot];
        @tensor hess_g[:,tot][-1;-2] += K_o_a[-1;1]*dK[-2;1];
        

        for (ind,(ai,bi)) in enumerate(two_inds)
            tohandle = [(As[ind],Bs[ind],(ai,bi))];
            if ai!=bi
                push!(tohandle,(As[ind],Bs[ind]',(bi,ai)));
            end

            for (cur_A,cur_B,(cur_I,cur_J)) in tohandle
                temp_V = @view dV[:,cur_I,cur_J,:];
                temp_A = (x*cur_A+cur_A*x')[:,tot]
                @tensor hess_g[:,tot][-1,-2] += 2*temp_A[-1,1]*temp_V[-2,1]

                temp_V = @view dV[:,cur_I,:,cur_J];
                temp_A = cur_B*x'[:,tot]
                @tensor hess_g[:,tot][-1,-2] += 2*temp_A[-1,1]*temp_V[-2,1]

                temp_V = @view dV[:,:,cur_I,cur_J];
                temp_A = cur_B*x'[:,tot]
                @tensor hess_g[:,tot][-1,-2] += 2*temp_A[-1,1]*temp_V[-2,1]

                temp_V = @view dV[cur_I,:,:,cur_J];
                temp_A = @view (x*cur_A+cur_A*x')[tot,:];
                @tensor hess_g[tot,:][-1,-2] -= 2*temp_A[1,-2]*temp_V[1,-1]

                temp_V = @view dV[cur_I,:,cur_J,:];
                temp_A = @view (x*cur_B)[tot,:];
                @tensor hess_g[tot,:][-1,-2] -= 2*temp_A[1,-2]*temp_V[1,-1]

                temp_V = @view dV[cur_I,cur_J,:,:];
                temp_A = @view (x*cur_B)[tot,:];
                @tensor hess_g[tot,:][-1,-2] -= 2*temp_A[1,-2]*temp_V[1,-1]
                
            end
        end

        #hess_g = (hess_g-hess_g')/2
        return -hess_g

    end


    -g,apply_hessian
end


function transform(h::CASSCF_Ham,ABs = get_A_B(h))
    B = h.ao2mo
    c_eri = zeros(eltype(B),length(h.active),length(h.active),length(h.active),length(h.active));
    c_k = zeros(eltype(B),length(h.active),length(h.active));

    V_nuc = GaussianBasis.Molecules.nuclear_repulsion(h.basis.atoms);

    K = B'*(nuclear(h.basis)+kinetic(h.basis))*B;
    c_k += K[h.active,h.active]

    filled = 1:(h.active.start-1);

    V_nuc += tr(K[filled,filled])*2
    
    (two_inds,As,Bs) = ABs

    for (ind,(ai,bi)) in enumerate(two_inds)
        tohandle = [(As[ind],Bs[ind],(ai,bi))];
        if ai!=bi
            push!(tohandle,(As[ind],Bs[ind]',(bi,ai)));
        end

        for (cur_A,cur_B,(cur_I,cur_J)) in tohandle
            if cur_I in h.active && cur_J in h.active
                c_eri[cur_I-h.active.start+1,:,:,cur_J-h.active.start+1] += cur_A[h.active,h.active]

                c_k[cur_I-h.active.start+1,cur_J-h.active.start+1]+=tr(cur_A[filled,filled])*4;
                c_k[cur_I-h.active.start+1,cur_J-h.active.start+1]-=tr(cur_B[filled,filled])*2;

            end

            if cur_I in filled && cur_I == cur_J
                V_nuc += 4*tr(cur_A[filled,filled])
                V_nuc -= 2*tr(cur_B[filled,filled])
            end

        end
    end
  
    return (V_nuc,c_k,c_eri)

end

function mpo_representation(h::CASSCF_Ham)
    (E,K,ERI) = transform(h);    
    fused_quantum_chemistry_hamiltonian(E,K,ERI,Float64)
end


function orb_retract(x,grad_U,alpha)
    #@show "retracting",alpha
    #flush(stderr); flush(stdout);
    (ham,dV,dK) = x;
    new_U = ham.ao2mo*exp(alpha*grad_U)'
 
    orbmanpoint(CASSCF_Ham(ham.basis,new_U,ham.active),dV,dK),grad_U
end


function get_A_B(h::CASSCF_Ham)
    B = h.ao2mo

    tot = 1:h.active.stop

    two_inds = Tuple{Int,Int}[];
    for i in tot, j in tot
        j >= i || continue;
        push!(two_inds,(i,j))
    end
    locks = map(x->SpinLock(),Iterators.product(1:h.basis.nshells,1:h.basis.nshells));

    As = [zeros(eltype(h.ao2mo),h.basis.nbas,h.basis.nbas) for i in two_inds]
    Bs = [zeros(eltype(h.ao2mo),h.basis.nbas,h.basis.nbas) for i in two_inds]

    Nvals = GaussianBasis.num_basis.(h.basis.basis)
    ao_offset = [sum(Nvals[1:(i-1)])  for i = 1:h.basis.nshells]  

    ijkls = Tuple{Int,Int,Int,Int}[];
    for i in 1:h.basis.nshells, j in 1:i, k in 1:i, l in 1:k
        k!=i || j>=l || continue; 
        push!(ijkls,(i,j,k,l))
    end


    @threads for (i,j,k,l) in ijkls

        buff = ERI_2e4c(h.basis,i,j,k,l)/2;
        norm(buff)<1e-12 && continue

        eightfold_way = unique([(1,2,3,4),(2,1,3,4),(2,1,4,3),(1,2,4,3),(3,4,1,2),(3,4,2,1),(4,3,2,1),(4,3,1,2)]) do (ai,bi,ci,di)
            (a,b,c,d) = map(x->(i,j,k,l)[x],(ai,bi,ci,di))
        end
        
        for (ai,bi,ci,di) in eightfold_way
            
            (a,b,c,d) = map(x->(i,j,k,l)[x],(ai,bi,ci,di))
            buff_perm = permutedims(buff,(ai,bi,ci,di));

            sl_a = ao_offset[a]+1:ao_offset[a]+Nvals[a];
            sl_b = ao_offset[b]+1:ao_offset[b]+Nvals[b];
            sl_c = ao_offset[c]+1:ao_offset[c]+Nvals[c];
            sl_d = ao_offset[d]+1:ao_offset[d]+Nvals[d];
            
            t_1 = @view B'[tot,sl_a];
            t_4 = @view B[sl_b,tot];
            @tensor t1[-1 -2;-3 -4] := t_1[-1,1]*t_4[4,-4]*buff_perm[1,4,-2,-3]

            lock(locks[c,d]) do
                # update As
                for (ind,(e,f)) in enumerate(two_inds)
                    As[ind][sl_c,sl_d] += t1[e,:,:,f]
                end
            end


            t_2 = @view B'[tot,sl_c];
            t_4 = @view B[sl_b,tot];
            @tensor t1[-1 -2;-3 -4] := t_2[-2,2]*t_4[4,-4]*buff_perm[-1,4,2,-3]

            lock(locks[a,d]) do
                for (ind,(e,f)) in enumerate(two_inds)
                    Bs[ind][sl_a,sl_d] += t1[:,e,:,f]
                end
            end
            
        end
    end
    for i in 1:length(As)
        As[i] = B'*As[i]*B
        Bs[i] = B'*Bs[i]*B
    end

    return (two_inds,As,Bs)

end

orb_inner(x, g1, g2) = real(dot(g1,g2))
orb_scale!(g, alpha) = g*alpha 
orb_add!(g1, g2, alpha) = g1+g2*alpha

function orb_cfun(x)
    (ham,dV,dK,ABs,(dU,hessian)) = x;

    (V_nuc,c_k,c_eri) = transform(ham,ABs);

    E = V_nuc;
    E += @tensor c_eri[1,2,3,4]*dV[1,2,3,4];
    E += @tensor c_k[1,2]*dK[1,2]

    real(E),copy(dU)
end


function orb_transport!(h, x, g, alpha, xp)
    ϵ = exp((alpha/2)*g)
    return ϵ*h*ϵ'
end


function orb_precondition(x,v)
    
    (ham,dV,dK,ABs,(grad,hessian)) = x;

    #=
    function precfun(w)
        (x_1,x_2) = w.vecs;
        x_1 = (x_1 - x_1')/2
        y_1 = hessian(x_1) + grad*x_2[1]
        y_2 = real(dot(grad,x_1))

        y_1 = (y_1-y_1')/2
        RecursiveVec(y_1,[y_2])
    end
    
    (vals,vecs,convhist) = eigsolve(precfun,RecursiveVec(v,[1]),1,:SR,Lanczos())
    (nsol,α)=first(vecs).vecs
    nsol =  nsol/α[1]
    nsol = (nsol-nsol')/2
    @show vals,convhist.normres, dot(grad,nsol)
    =#
    (nsol,convhist) = linsolve(hessian,v,v,GMRES(tol=norm(grad)/10))
    #@show convhist.normres    

    nsol = (nsol-nsol')/2

    if real(dot(grad,nsol))>0
        return nsol
    else
        return v
    end

    return -real.(nsol)
end

