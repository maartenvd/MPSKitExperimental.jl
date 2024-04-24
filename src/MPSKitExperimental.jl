module MPSKitExperimental
    using TensorKit,MPSKit,TensorOperations,KrylovKit,Strided, OptimKit, TensorKitManifolds
    using FLoops,Transducers,FoldsThreads, ConcurrentCollections
    using Base.Threads, LinearAlgebra

    using JLD2


    # stolen from unregistered https://github.com/lkdvos/AllocationKit.jl/blob/master/src/malloc.jl
    const MallocBackend = TensorOperations.Backend{:malloc}

    function TensorOperations.tensoralloc(::Type{Array{T,N}}, structure, istemp, ::MallocBackend) where {T,N}
        if istemp
            @assert isbitstype(T)
            ptr = Base.Libc.malloc(prod(structure) * sizeof(T))
            return unsafe_wrap(Array, convert(Ptr{T}, ptr), structure)
        else
            return tensoralloc(Array{T,N}, structure, istemp)
        end
    end

    function TensorOperations.tensorfree!(t::Array, ::MallocBackend)
        Base.Libc.free(pointer(t))
        return nothing
    end

    const SafeMallocBackend = TensorOperations.Backend{:safemalloc}

    function TensorOperations.tensoralloc(::Type{Array{T,N}}, structure, istemp, ::SafeMallocBackend) where {T,N}
        if istemp
            @assert isbitstype(T)
            ptr = Base.Libc.malloc(prod(structure) * sizeof(T))
            A = unsafe_wrap(Array, convert(Ptr{T}, ptr), structure)
            finalizer(Base.Fix2(TensorOperations.tensorfree!, MallocBackend()), A)
        else
            return tensoralloc(Array{T,N}, structure, istemp)
        end
    end

    function TensorOperations.tensorfree!(t::Array, ::SafeMallocBackend)
        finalize(t)
        return nothing
    end

    export LeftGaugedMW, AssymptoticScatter,extend,partialdot,s_proj,projdown
    include("momentumwindow/momentum_window.jl")
    include("momentumwindow/orthoview.jl")
    include("momentumwindow/excitransfers.jl")
    include("momentumwindow/fusing_transfermatrix.jl")
    include("momentumwindow/assymptotic.jl")
    include("momentumwindow/effective_ex.jl")
    include("momentumwindow/timestep.jl")
    include("momentumwindow/mpo_envs.jl")
    include("momentumwindow/find_groundstate.jl")

    export @tightloop_tensor,@tightloop_planar
    include("tightloop/symbolic.jl")
    include("tightloop/tightloop.jl")
    include("tightloop/tensoroperations.jl")
    include("tightloop/planar.jl")
    
    export parse_fcidump, fused_quantum_chemistry_hamiltonian, disk_environments
    using MPSKit:fill_data!
    # contains most of the "tricks" needed to avoid tensorkit bottlenecks. 
    # You can play with these files to make them fall back to the default tensorkit implementation
    include("quantumchemistry/delayed_factory.jl");
    include("quantumchemistry/transpose_factory.jl")

    # fused_mpoham is a new type of mpohamiltonian, that allows for a "blocking" step
    # I also needed environments - derivatives for this new mpohamiltonian
    include("quantumchemistry/fused_mpoham.jl");
    include("quantumchemistry/fused_env.jl");
    include("quantumchemistry/fused_deriv.jl");

    # implements the qchem hamiltonian as a fused_mpoham
    include("quantumchemistry/qchem_operator.jl");
    include("quantumchemistry/compress.jl"); # minimal compressing step, which removes a bunch of exact zeros, by making the mpoham-bond dimension site dependent
    
    include("quantumchemistry/fcidump_parser.jl"); # simple parser for fcidump files

    # diskmanager uses a memory mapped file to store/transfer objects to. This automatically gives async IO
    include("quantumchemistry/diskmanager.jl")
    include("quantumchemistry/disk_backed_envs.jl")
    include("quantumchemistry/disk_backed_envs_manual.jl") # alternative to the diskmanager is to manually write data to disk
    
    using MPSKit:GrassmannMPS
    using GaussianBasis
    export CASSCF_Ham, GrassmannSCF;
    include("quantumchemistry/grassmann_scf.jl")
    include("quantumchemistry/orbopt.jl")
    

end
