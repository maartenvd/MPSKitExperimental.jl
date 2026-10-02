# Fair comparison: fused MPO vs MPSKit for quantum chemistry

These scripts compare DMRG2 on the N₂ FCIDUMP examples between:

| Name | MPO | Format / code path |
| --- | --- | --- |
| `fused` | `quantum_chemistry_hamiltonian` | `LinkMPOHamiltonian` (this package): channel operators on the sites, scalar link matrices |
| `fused_norr` | same | same, without the rank reduction (QR per pair of subblock keys) in `link_AC2_hamiltonian` |
| `fused_jordan` | same | `FiniteMPOHamiltonian(fused)`: the same operator on the same bond states, as MPSKit `JordanMPOTensor`s |
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

N₂/cc-pVDZ (28 orbitals), D = 50, 1 thread, warm, load average ~3. Measured 2026-10-03 with `LinkMPOHamiltonian`
(per-key-pair rank reduction in `link_AC2_hamiltonian`, environment basis derived from the links). `fused_jordan` runs on
the same bond states as `fused`, so it benefits from the smaller derived basis too.

**Full DMRG2 sweep** (`sweeps.jl`, one warm sweep from the reference state; all variants reach E = -108.7526401832):

| Variant | Time per sweep | Relative to `fused_jordan` | Previous code (FusedMPOHamiltonian) |
| --- | --- | --- | --- |
| `fused_jordan` | 15.9 s | 1.00× | 20.9 s |
| `fused` | 17.1 s | 1.07× | 25.0 s |
| `fused_norr` | 17.6 s | 1.11× | 21.1 s |
| `opsum` | 100.0 s | 6.28× | 132.8 s |

**Kernels** (`kernels.jl`, minimum in ms; bond 15 holds the NC/CN switch, the dense block of V in the link):

| Bond | Case | `opsum` | `fused_jordan` | `fused` | `fused` (rank red.) |
| --- | --- | --- | --- | --- | --- |
| 7 | build | 374 | 99 | 72 | 75 |
| 7 | apply | 3.9 | 1.7 | 2.3 | 2.2 |
| 7 | transfer_left | 41 | 10 | 14 | — |
| 14 | build | 9,957 | 765 | 639 | 828 |
| 14 | apply | 15.5 | 5.2 | 17.4 | 16.8 |
| 14 | transfer_left | 144 | 32 | 33 | — |
| 15 | build | 9,602 | 590 | 310 | 348 |
| 15 | apply | 15.1 | 5.1 | 10.4 | 10.2 |
| 15 | transfer_left | 2,079 | 117 | 69 | — |
| 21 | build | 473 | 102 | 112 | 112 |
| 21 | apply | 5.1 | 2.1 | 3.7 | 3.5 |
| 21 | transfer_left | 80 | 15 | 27 | — |

**MPO size** (`setup.jl`; bond states / total dimension of the bond to the right of the site, and genuine operator entries,
or channels for `fused`):

| Site | `fused` | `fused_jordan` | `opsum` |
| --- | --- | --- | --- |
| 7 | 128 / 478, 142 channels | 128 / 478, 101 entries | 737 / 1451, 475 entries |
| 14 | 421 / 1678, 436 channels | 421 / 1678, 214 entries | 2984 / 5924, 1090 entries |
| 15 | 381 / 1462, 853 channels | 381 / 1462, 13,364 entries | 2619 / 5197, 353,158 entries |
| 21 | 135 / 478, 264 channels | 135 / 478, 455 entries | 765 / 1507, 4,577 entries |

Conclusions:

- **The speedup over the OpSum MPO is the MPO construction, not the format** (unchanged from the previous code): on the
  same operator, MPSKit's `JordanMPOTensor` path is within 7% of the link code per sweep. The OpSum MPO is 3–4× too large,
  see lkdvos/OpSum.jl#38.
- **The scalar switch pays off where it should:** at bond 15, where the dense block of V is a scalar link instead of
  13,364 operator entries, the link code builds the AC2 Hamiltonian 1.9× and grows environments 1.7× faster.
- **Elsewhere MPSKit is faster:** its AC2 apply is 1.3–3.3× faster at every bond, and its transfers 1.4–1.8× faster at the
  ordinary bonds. The apply is called many times per eigensolve, which is why `fused_jordan` wins the sweep overall.
- **Rank reduction now helps a little** (`fused` 17.1 s vs `fused_norr` 17.6 s); in the previous code it cost 16%.
- Both formats got faster than before (20.9 → 15.9 s, 25.0 → 17.1 s): the environment basis derived from the links has
  fewer bond states than the hand-built one (4510 vs 4973 summed over bonds).
