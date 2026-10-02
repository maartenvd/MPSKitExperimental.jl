module MPSKitExperimental
    using TensorKit,MPSKit,TensorOperations,KrylovKit,Strided, OptimKit, TensorKitManifolds
    using FLoops,Transducers,FoldsThreads, ConcurrentCollections
    using Base.Threads, LinearAlgebra, SparseArrays

    using JLD2
    using MPSKit:MPSTensor,MPSBondTensor,MPOTensor,_firstspace,_lastspace,_transpose_tail,_transpose_front,Multiline,LeftGaugedQP;

    #_firstspace(t::AbstractTensorMap) = space(t, 1)
    #_lastspace(t::AbstractTensorMap) = space(t, numind(t))


    # stolen from unregistered https://github.com/lkdvos/AllocationKit.jl/blob/master/src/malloc.jl
    
    # temporaries (istemp = Val(true)) are malloc'ed and must be released with tensorfree!;
    # thread safe and needs no sizing, unlike TensorOperations.BufferAllocator.
    # `malloc` is an instance, so it can be passed directly as `allocator=malloc`.
    struct MallocBackend end
    const malloc = MallocBackend()
    (m::MallocBackend)() = m # `allocator = malloc()` also works
    export malloc

    # number of malloc'ed temporaries that have not been freed yet
    const leak_counter = Threads.Atomic{Int}(0)

    function _malloc_array(::Type{T}, structure) where {T}
        isbitstype(T) || throw(ArgumentError("malloc allocator requires an isbits element type, got $T"))
        ptr = Base.Libc.malloc(max(prod(structure), 1) * sizeof(T)) # malloc(0) may return NULL
        ptr == C_NULL && throw(OutOfMemoryError())
        return unsafe_wrap(Array, convert(Ptr{T}, ptr), structure)
    end

    TensorOperations.tensoralloc(::Type{Array{T,N}}, structure, ::Val{false}, ::MallocBackend) where {T,N} =
        tensoralloc(Array{T,N}, structure, Val(false))
    function TensorOperations.tensoralloc(::Type{Array{T,N}}, structure, ::Val{true}, ::MallocBackend) where {T,N}
        A = _malloc_array(T, structure)
        atomic_add!(leak_counter,1)
        return A
    end

    function TensorOperations.tensorfree!(t::Array, ::MallocBackend)
        atomic_add!(leak_counter,-1)
        Base.Libc.free(pointer(t))
        return nothing
    end
    # TensorOperations frees the parent of a StridedView, which is the array's `Memory`
    # rather than the `Array` itself
    @static if isdefined(Core, :Memory)
        # TensorKit's transform kernel allocates its recoupling buffers as `Memory`; without
        # this method TensorOperations falls back to GC memory, which tensorfree! would then free
        TensorOperations.tensoralloc(::Type{Memory{T}}, structure, ::Val{false}, ::MallocBackend) where {T} =
            tensoralloc(Memory{T}, structure, Val(false))
        function TensorOperations.tensoralloc(::Type{Memory{T}}, structure, ::Val{true}, ::MallocBackend) where {T}
            A = _malloc_array(T, prod(structure))
            atomic_add!(leak_counter,1)
            return unsafe_wrap(Memory{T}, pointer(A), length(A))
        end
        function TensorOperations.tensorfree!(t::Memory, ::MallocBackend)
            atomic_add!(leak_counter,-1)
            Base.Libc.free(pointer(t))
            return nothing
        end
    end

    # malloc'ed temporaries owned by julia: released by the GC, never by tensorfree!
    struct SafeMallocBackend end

    TensorOperations.tensoralloc(::Type{Array{T,N}}, structure, ::Val{false}, ::SafeMallocBackend) where {T,N} =
        tensoralloc(Array{T,N}, structure, Val(false))
    function TensorOperations.tensoralloc(::Type{Array{T,N}}, structure, ::Val{true}, ::SafeMallocBackend) where {T,N}
        A = _malloc_array(T, structure)
        return unsafe_wrap(Array, pointer(A), structure; own = true)
    end
    using MPSKit:TransferMatrix,auxiliaryspace
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
    export variance_environments, variance_proj
    include("momentumwindow/variance.jl")
    export scatter_lsq!, scatter_galerkin!
    include("momentumwindow/scattering.jl")

    export @tightloop_tensor,@tightloop_planar
    include("tightloop/symbolic.jl")
    include("tightloop/tightloop.jl")
    include("tightloop/tensoroperations.jl")
    include("tightloop/planar.jl")
    
    export parse_fcidump, quantum_chemistry_hamiltonian, disk_environments
    using MPSKit:fill_data!, add_util_leg, l_LL, r_RR
    
    # a hamiltonian as on-site operators + scalar link matrices, with its environments and derivatives
    export LinkMPOHamiltonian, link_channels, link_gradient
    include("quantumchemistry/link_mpoham.jl");
    include("quantumchemistry/link_env.jl");
    include("quantumchemistry/link_deriv.jl");
    include("quantumchemistry/link_gradient.jl");

    # the qchem hamiltonian, built once per number of orbitals with symbolic integrals
    export qchem_structure
    include("quantumchemistry/qchem_operator.jl");
    include("quantumchemistry/jordan_conversion.jl"); # FiniteMPOHamiltonian(::LinkMPOHamiltonian), for comparisons with MPSKit
    
    include("quantumchemistry/fcidump_parser.jl"); # simple parser for fcidump files

    # diskmanager uses a memory mapped file to store/transfer objects to. This automatically gives async IO
    #include("quantumchemistry/diskmanager.jl")
    include("quantumchemistry/disk_backed_envs_manual.jl") # alternative to the diskmanager is to manually write data to disk
    
    # orbital optimization (co-optimized with the mps, or alternating with dmrg)
    using MPSKit: GrassmannMPS
    import TensorKitManifolds.Grassmann
    export QChemIntegrals, GrassmannSCF, DMRGSCF, qchem_rdms, optimize_orbitals, rotate_integrals, rdm_energy, qchem_mpo, active_space, embed_rdms
    include("quantumchemistry/grassmann_scf.jl")
    
    export tight_AC_hamiltonian
    include("fastmpoham/contractions.jl")
    include("fastmpoham/fastmpoham.jl")

end
