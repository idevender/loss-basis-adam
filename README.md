# The Loss Does Not See the Basis, but Adam Does

Code for the paper of the same name, by Devender Singh.

**Paper:** [arXiv:2608.05136](https://arxiv.org/abs/2608.05136) — **Project page:**
[idevender.github.io](https://idevender.github.io/) — **Questions:** devenders@mun.ca

A factored loss `L(UV^T)` does not change when you rotate both factors: `(U, V) -> (UQ, VQ)` for
orthogonal `Q`. Gradient descent respects that symmetry; Adam does not. The experiments here test
what follows. Equivariant methods (GD, momentum, shared-scalar Adam, Muon, Shampoo) keep gradient
descent's low-rank bias. Coordinate-wise methods (Adam, RMSProp, Lion, signum, Adafactor) follow
the arbitrary basis of the factors instead. The same symmetry turns up in attention heads, and it
motivates a gauge-preserving variant of FlowAdam.

## Install

```bash
git clone https://github.com/idevender/loss-basis-adam.git
cd loss-basis-adam
pip install -e .
pip install -r requirements.txt
```

Python 3.9+. Each CPU experiment takes a few laptop-minutes. The replication ladder in
`experiments/nibi/` needs a GPU.

## Start here

The headline result — Figure 1 and Table 2, the optimizer zoo — runs on a laptop:

```bash
cd experiments
python optimizer_zoo_bias.py
```

Every method runs to interpolation, so what separates them is which solution gets selected, not how
far each one got. Per-`(method, lr, seed)` records land in `logs/optimizer_zoo_bias.jsonl`, which
also holds the selection used by the paper. The rest of the paper's scripts are tabulated under
[Reproducing the paper](#reproducing-the-paper).

## Layout

```
flowadam/            the optimizer package
experiments/         CPU experiments
experiments/nibi/    H100 replication ladder
logs/                stdout and JSONL from the CPU runs
figures/             paper-figure generation
tests/               numerical checks of the paper's identities
```

## The optimizer

`flowadam/optimizer.py` implements FlowAdam. Two knobs matter for the paper:

- **`precond_power`** — the anisotropy dial `p`. The denominator is `s_i^p * sbar^(1-p)`, where
  `s_i = sqrt(vhat_i)` and `sbar` is a shared scalar. `p=1` is ordinary per-coordinate Adam;
  `p=0` is the gauge-invariant end.
- **`precond_scalar`** — `'rms'` or `'geomean'`. Only `'rms'` is exactly invariant at `p=0`.

```python
import torch
from flowadam import FlowAdam

model = torch.nn.Linear(64, 64)
x = torch.randn(8, 64)
opt = FlowAdam(model.parameters(), lr=1e-3,
               precond_power=0.0, precond_scalar='rms',
               clip_mode='globalnorm', clip_norm_c=10.0)

def closure():
    opt.zero_grad()
    loss = model(x).pow(2).mean()
    loss.backward()
    return loss

opt.step(closure)
```

An exactly gauge-invariant `p=0` run needs `precond_scalar='rms'` **and**
`clip_mode='globalnorm'`. The default per-coordinate clip breaks the gauge on any step where the
ODE path fires, and the package still defaults to `'geomean'` for backwards compatibility. FlowAdam
takes one parameter group and requires a closure.

## Tests

```bash
pytest -q tests/test_paper_identities.py      # 25 tests, 41 parametrized runs
```

Each identity is evaluated on both sides in float64 at random points, with a real loss, real
autograd gradients, and the update rules imported from the experiment code that produced the
tables. Every positive claim is paired with a negative control that has to fail. Test names follow
the manuscript's numbering, e.g. `test_thm46_transfer_theorem_time_change_direction` for
Theorem 4.6.

## Reproducing the paper

Run from `experiments/`, where the sibling imports resolve:

```bash
cd experiments
python optimizer_zoo_bias.py
```

| Script | Paper | What it shows |
| --- | --- | --- |
| `optimizer_zoo_bias.py` | §5 | The zoo map: equivariant vs. coordinate-wise recovery |
| `zoo_lr_sensitivity.py` | §5, D.2 | Learning-rate curves for the zoo |
| `zoo_decay_control.py` | D.1 | The split survives cosine-for-all |
| `zoo_init_scale.py` | D.4 | The split survives init scale |
| `zoo_size_check.py` | §5 | The ordering survives a second size and rank |
| `nuclear_norm_reference.py` | §5, Table 2 | Min-nuclear-norm interpolant, the reference row |
| `precond_dial_scalar_check.py` | §7, Fig. 3 | The dial `p: 1 -> 0` under both scalars |
| `precond_dial_fixed_lr.py` | §7, App. C | The dial's fixed-step arm across the shared lr grid |
| `equivariance_balance_probe.py` | §7 | Gauge product-drift and balancedness |
| `muon_shampoo_bias_probe.py` | §5 | Muon and Shampoo keep the bias (geometry, not balancedness) |
| `muon_phase_diagram.py` | §8 | Spectral-tail phase diagram and the crossing |
| `phase_diagram_decay_control.py` | D.1 | Schedule control for the phase diagram |
| `signum_c9_audit.py` | D.10 | Recovery without equivariance: annealed sign descent |
| `attention_gauge.py` | §6 | Attention-gauge onset: Adam splits after one step |
| `attention_gauge_multiseed.py` | D.5 | The same across seeds, gauge draws and noise scales |
| `hyperspectral_wilson_v3.py` | §9, D.6 | Indian Pines on CPU: matched loss, train-only lr selection |
| `flowadam_upgrade.py` | §10 | The two clip modes, plus the legacy geometric-mean dial |
| `flowadam_upgrade_rms.py` | §10 | FlowAdam-p under the RMS scalar; the source of the 0.169 |
| `flowadam_p_interp_control.py` | D.8 | Interpolation control, not an early-stopping artifact |
| `restoration_probe.py` | shared | The matrix-sensing testbed used across the above |

`hyperspectral_wilson_v3.py` is the CPU real-data protocol. `hyperspectral_completion.py` and
`hyperspectral_wilson_v2.py` are earlier variants, kept for loader coverage rather than results.

### The H100 ladder (Appendix D.9)

`experiments/nibi/` is a self-contained GPU suite: device- and dtype-agnostic ports of the CPU
protocols. Run the drivers, then aggregate.

```bash
cd experiments/nibi
mkdir -p ../nibi_results
python zoo_ladder.py --n 40 --out ../nibi_results/zoo_n40.jsonl
python phase_fine.py --out ../nibi_results/phase_n40.jsonl
python dial_scale.py --out ../nibi_results/dial_n40.jsonl --flowadam
python attention_suite.py --task mod --out ../nibi_results/attn_mod.jsonl
python attention_suite.py --task text --depth 6 --dmodel 256 \
  --data ../../data/text/input.txt --out ../nibi_results/attn_text.jsonl
python pavia.py --dataset paviau --mat /path/to/PaviaU.mat \
  --densities 0.28,0.46 --out ../nibi_results/pavia_paviau.jsonl
python collect.py --dir ../nibi_results --md ../nibi_results/SUMMARY.md
```

Other ladder sizes go to their own files (`--n 128 --out ../nibi_results/zoo_n128.jsonl`). For
Indian Pines, use `--dataset indianpines --densities 0.15,0.25`.

### Figures

```bash
cd figures
python make_figures.py               # Figures 1, 2, 3, 4
python make_realdata_trajectory.py   # Figure 5
```

Both write the vector PDFs the manuscript includes. Figure 5 is separate because it reads the
committed GPU records rather than a local run.

Figures 1 and 5 are recomputed rather than transcribed: Figure 1 redoes Table 2's learning-rate
selection from `logs/optimizer_zoo_bias.jsonl`, and Figure 5 redoes Appendix D.6's train-only rule
from the GPU records, selecting on seeds 42 and 123. Both assert they land on the published
numbers, so neither can drift without failing.

The attention-gauge curves are a snapshot of `logs/attention_gauge_cpu.txt` and
`logs/attention_gauge_noise_cpu.txt`. Regenerate those logs and the figure together.

## Raw records

`logs/` holds the CPU output behind Section 5 and the appendix controls.

| File | Backs |
| --- | --- |
| `optimizer_zoo_bias.jsonl` / `.txt` | Every cell of Table 2 and every bar of Figure 1, selection included: one record per (method, lr, seed), plus one `selected` record per method |
| `nuclear_norm_reference.txt` | Table 2's convex reference row |
| `zoo_lr_sensitivity.txt`, `zoo_decay_control.txt`, `zoo_init_scale.txt`, `zoo_size_check.txt` | The Appendix D controls that row rests on |
| `signum_c9_audit_cpu.txt` | Appendix D.10: the horizon × schedule grid, the rank-one falsification, the spectral-tail sweep |
| `attention_gauge_multiseed_cpu.txt` | Appendix D.5: six Adam gauge pairs, noise twins at both scales, the equivariant and Muon controls. Prints the 662× structural ratio directly |
| `precond_dial_scalar_check_cpu.txt` | Both dial arms, RMS and geometric-mean, behind Section 7 |
| `flowadam_upgrade_cpu.txt` | The Section 10 sweep (see the note below) |
| `muon_phase_diagram_cpu.txt`, `phase_diagram_decay_control_cpu.txt` | Section 8's three-seed tail sweep and its cosine-decay control |

Two of those logs will not line up with the paper unless you know why:

- **`flowadam_upgrade_cpu.txt`.** Its `best()` scores a non-interpolating config as `rec + 10`, so
  each row is the best *interpolating* learning rate, not the fixed 1e-3 anchor Section 10 quotes.
  It backs the per-coordinate reading (0.3466 vs. the paper's 0.347) and the geometric-mean
  `Adam-p=0` endpoint (0.2292, matching the dial log), but not Section 10's 0.220 and 0.169, which
  are deep fits at 6e-7 and 7e-6. Its closing `-17.4%` compares the two selected rows, not the
  fixed-rate pair.
- **`dial_flowlong_n40.txt`.** This continues the finer-grid n=40 FlowAdam-p cell, the one that
  clears the 1e-7 bar only past the sweep's 30k steps — not the 0.169. All ten seeds interpolate
  by 6e4 steps and mean 0.1475 +- 0.0357, the value Section 10 quotes. (Section 10's other
  extended-budget number, 0.1694 -> 0.1691, is the separate 120k-step early-stopping control from
  `flowadam_p_interp_control.py`, Appendix D.8.)
- **`muon_phase_diagram_cpu.txt`.** Its three-seed crossing sits at τ*≈0.35; the paper's τ*≈0.2 is
  the sharper ten-seed estimate, as Appendix D notes.

`experiments/nibi_results/` holds the H100 JSONL behind Table 5 (GPU hyperspectral and Pavia),
Table 9 (twin drift at scale) and Table 11 (the problem-size ladder). Those runs need a GPU
allocation, so the records are committed and the tables stay checkable without one — including
`python collect.py --dir ../nibi_results`, which regenerates `SUMMARY.md`. Each `zoo_n*.jsonl` has
one `select` record naming the learning rate chosen per method, plus one record per
(method, lr, seed); Table 11's cells are the mean `rec` over seeds at the selected rate.

One file is not derivable from the others. `dial_n40_flowlong.jsonl` is the extended-budget rerun
of the single dial cell that hit the step cap without interpolating, written by
`experiments/nibi/dial_flowlong.py`. Collecting over `dial_n40.jsonl` alone will not reproduce it.

## Data

The synthetic sensing tasks are generated in code. The real-data experiments use two public
hyperspectral scenes and a character-level corpus, none of them shipped here:

- **Indian Pines** (`Indian_pines_corrected.mat`, key `indian_pines_corrected`) and **Pavia
  University** (`PaviaU.mat`, key `paviaU`), both from the hyperspectral remote-sensing scenes
  collection at the University of the Basque Country (ehu.eus). Put `Indian_pines_corrected.mat`
  under `data/hyperspectral/` for the CPU scripts, or pass `--mat` to `experiments/nibi/pavia.py`.
- **tiny-Shakespeare** for the character-LM attention twins (`input.txt`, ~1.1 MB, from
  `data/tinyshakespeare/input.txt` in github.com/karpathy/char-rnn). Put it at `data/text/input.txt`
  or pass `--data`. The paper's char-LM row uses `--task text --depth 6 --dmodel 256`, everything
  else at defaults.

## Citation

```bibtex
@article{singh2026lossbasis,
  title         = {The Loss Does Not See the Basis, but Adam Does},
  author        = {Singh, Devender},
  year          = {2026},
  eprint        = {2608.05136},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2608.05136}
}
```

## License

MIT. See [LICENSE](LICENSE).
