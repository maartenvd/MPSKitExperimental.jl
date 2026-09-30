# tightloop-compiled kernels for the additive terms of MPSKit's JordanMPO_AC_Hamiltonian
# apply step (see fastmpoham.jl and MPSKit.jl's algorithms/derivatives/hamiltonian_derivatives.jl):
#   y = A(x); y += x*D; y += E*x; y += x*I; y += x*C; y += B*x
# MPSKit writes these with @plansor, which uses the (non-planar) @tensor path for Bosonic
# sectors and the @planar path otherwise, so both variants are compiled here and
# tight_AC_hamiltonian picks one the same way.
# allocator=malloc: thread safe (the factories are shared between all sites and threads,
# a BufferAllocator created at macro expansion time would be shared as well) and free of GC.
@tightloop_planar tight_D_apply_planar allocator=malloc y[-1 -2;-3] += x[-1 1;-3]*D[-2;1]
@tightloop_planar tight_E_apply_planar allocator=malloc y[-1 -2;-3] += E[-1;1]*x[1 -2;-3]
@tightloop_planar tight_I_apply_planar allocator=malloc y[-1 -2;-3] += x[-1 -2;1]*I[1;-3]
@tightloop_planar tight_C_apply_planar allocator=malloc y[-1 -2;-3] += x[-1 2;1]*C[-2 -3;2 1]
@tightloop_planar tight_B_apply_planar allocator=malloc y[-1 -2;-3] += B[-1 -2;1 2]*x[1 2;-3]

@tightloop_tensor tight_D_apply_tensor allocator=malloc y[-1 -2;-3] += x[-1 1;-3]*D[-2;1]
@tightloop_tensor tight_E_apply_tensor allocator=malloc y[-1 -2;-3] += E[-1;1]*x[1 -2;-3]
@tightloop_tensor tight_I_apply_tensor allocator=malloc y[-1 -2;-3] += x[-1 -2;1]*I[1;-3]
@tightloop_tensor tight_C_apply_tensor allocator=malloc y[-1 -2;-3] += x[-1 2;1]*C[-2 -3;2 1]
@tightloop_tensor tight_B_apply_tensor allocator=malloc y[-1 -2;-3] += B[-1 -2;1 2]*x[1 2;-3]
