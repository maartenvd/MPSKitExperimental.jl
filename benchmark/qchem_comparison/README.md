# Fair comparison: fused MPO vs MPSKit for quantum chemistry

These scripts compare DMRG2 on the N₂ FCIDUMP examples between:

| Name | MPO | Format / code path |
| --- | --- | --- |
| `fused` | `fused_quantum_chemistry_hamiltonian` | `FusedMPOHamiltonian` (this package) |
| `fused_norr` | same | same, without the QR/LQ rank reduction in `fused_AC2_hamiltonian` |
| `fused_jordan` | same | `FiniteMPOHamiltonian(fused)`: the same channels as MPSKit `JordanMPOTensor`s |
| `opsum` | `examples/opsum_qchem_su2.jl` (OpSum `BipartiteAlgorithm`) | MPSKit `JordanMPOTensor`s, converted to real scalars |
| `opsum_complex` | same | as OpSum emits it (`ComplexF64`), as in `examples/is_fusempo_nuttig.ipynb` |

`fused` vs `fused_jordan` isolates the storage format, and `fused_jordan` vs `opsum` isolates the MPO construction.

## Running

Julia 1.11 or 1.12 (`ConcurrentCollections`, a dependency of this package, does not build on 1.13).

```sh
cd benchmark/qchem_comparison
julia --project=. -e 'using Pkg; Pkg.instantiate()'
FCIDUMP=N2.STO3G.FCIDUMP julia --project=. setup.jl    # MPOs, checks, MPO summary, reference state
FCIDUMP=N2.STO3G.FCIDUMP julia --project=. kernels.jl  # build / apply / transfer per bond
FCIDUMP=N2.STO3G.FCIDUMP julia --project=. sweeps.jl   # full DMRG2 sweeps
```

`setup.jl` writes `data/<system>_D<bond>.jls`, which the other two scripts read. Settings, all through environment variables:

| Variable | Default | Used by | Meaning |
| --- | --- | --- | --- |
| `FCIDUMP` | `N2.STO3G.FCIDUMP` | all | integrals in `examples/` (`N2.CCPVDZ.FCIDUMP` for the 28-orbital case) |
| `BOND` | `50` | all | MPS bond dimension |
| `BLAS_THREADS` | `1` | all | BLAS threads; the julia thread count is set with `julia -t` |
| `SWEEPS` | `3` | setup | DMRG2 sweeps for the reference state |
| `MPOS` | see script | kernels, sweeps | which variants to compare |
| `POS` | central bond | kernels | bonds to time, e.g. `"7 14 21"` |
| `ROUNDS`, `BUDGET` | `3`, `2` | kernels | interleaved rounds, seconds per case per round |
| `REPEATS`, `WARMUP`, `TIMERS` | `1`, `1`, `0` | sweeps | timed sweeps per variant, untimed warm-up sweep, print MPSKit's timer |

## Pitfalls these scripts avoid

- **Different MPOs.** The OpSum MPO has a ~3.4× larger bond dimension than the fused one for cc-pVDZ, so comparing `opsum` against `fused` mixes the MPO construction with the storage format. `fused_jordan` puts the fused MPO into MPSKit's format.
- **Complex vs real.** OpSum always emits `ComplexF64`; `realify` converts it, keeping identity blocks as identity scalars.
- **Compilation.** The first sweep in a process includes compilation, which inflated earlier fused timings about 2×. `sweeps.jl` runs an untimed warm-up sweep first.
- **Shared machines.** Kernel timings are the minimum over interleaved rounds, and every run prints the load average.
- **Threading.** Defaults are 1 julia thread and 1 BLAS thread. The fused code threads over julia threads; MPSKit's DMRG2 sweep does not.

## Results

N₂/cc-pVDZ (28 orbitals), D = 50, 1 thread, warm. Measured 2026-10-02 on a shared workstation (load average 4–20), with the scripts as committed.

**Full DMRG2 sweep** (`sweeps.jl`, one warm sweep from the reference state):

| Variant | Time per sweep | Relative to `fused_jordan` |
| --- | --- | --- |
| `fused_jordan` | 20.9 s | 1.00× |
| `fused_norr` | 21.1 s | 1.01× |
| `fused` | 25.0 s | 1.20× |
| `opsum` | 132.8 s | 6.36× |

**Kernels** (`kernels.jl`, minimum in ms):

| Bond | Case | `opsum` | `fused_jordan` | `fused` | `fused` (rank red.) |
| --- | --- | --- | --- | --- | --- |
| 7 | build | 744 | 190 | 146 | 175 |
| 7 | apply | 5.3 | 3.1 | 2.8 | 4.4 |
| 7 | transfer_left | 54 | 13 | 27 | — |
| 14 | build | 13,206 | 932 | 847 | 1,240 |
| 14 | apply | 17.1 | 5.1 | 14.1 | 34.5 |
| 14 | transfer_left | 173 | 34 | 103 | — |
| 21 | build | 876 | 226 | 237 | 270 |
| 21 | apply | 4.6 | 3.0 | 4.4 | 4.8 |
| 21 | transfer_left | 101 | 21 | 57 | — |

**MPO size** (`setup.jl`; channels / total dimension of the bond to the right of the site, and genuine operator entries, or blocks for `fused`):

| Site | `fused` | `fused_jordan` | `opsum` |
| --- | --- | --- | --- |
| 7 | 128 / 478, 140 blocks | 128 / 478, 121 entries | 737 / 1451, 475 entries |
| 14 | 450 / 1738, 462 blocks | 450 / 1738, 204 entries | 2984 / 5924, 1090 entries |
| 15 | 409 / 1518, 879 blocks | 409 / 1518, 13,412 entries | 2619 / 5197, 353,158 entries |
| 21 | 163 / 534, 290 blocks | 163 / 534, 455 entries | 765 / 1507, 4,577 entries |

Conclusions:

- **The speedup over the OpSum MPO is the MPO construction, not the format:** on the same MPO, MPSKit's `JordanMPOTensor` path (`fused_jordan`) is as fast as the fused code or faster. The fused format only builds the AC2 Hamiltonian somewhat faster at some bonds, while MPSKit's transfers are 2–3× faster.
- **The QR/LQ rank reduction in `fused_AC2_hamiltonian` costs time:** `fused_norr` is 16% faster per sweep than `fused`.
- **The OpSum MPO is 3–4× too large:** see lkdvos/OpSum.jl#38.
