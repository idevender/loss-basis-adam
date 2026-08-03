# The Loss Does Not See the Basis, but Adam Does

Code for the paper **"The Loss Does Not See the Basis, but Adam Does"** (Devender Singh).

A factored loss `L(UV^T)` is invariant under the gauge action `(U, V) -> (UQ, VQ)` for
orthogonal `Q`. This repository contains experiments assessing whether an optimizer's implicit
bias on such models is related to equivariance under this action. In the reported protocols,
gauge-equivariant methods (gradient descent, momentum, shared-scalar Adam, Muon, Shampoo)
preserve gradient descent's low-rank bias, whereas coordinate-wise methods (Adam, RMSProp, Lion,
signum, Adafactor) depend on the arbitrary factor basis. The same symmetry appears in attention
heads and motivates a gauge-preserving variant of FlowAdam.

## Installation

```bash
git clone https://github.com/idevender/loss-basis-adam.git
cd loss-basis-adam
pip install -e .        # installs the flowadam package
pip install -r requirements.txt
```

Python 3.9+. CPU experiments run in minutes each. The replication ladder
(`experiments/nibi/`) needs a GPU.

## Tests

`tests/test_paper_identities.py` checks the paper's constructive identities numerically. Both
sides are evaluated in float64 at random points, using a real loss, real autograd gradients, and
the update rules imported from the experiment code that produced the tables (Muon's
Newton–Schulz iteration, Shampoo's inverse roots). Each positive claim is paired with a negative
control that must fail the same check.

```bash
pytest -q tests/test_paper_identities.py      # 25 tests, 41 parametrized runs
```

Test names follow the manuscript's numbering, for example
`test_thm46_transfer_theorem_time_change_direction` for Theorem 4.6.

## The optimizer

`flowadam/optimizer.py` implements FlowAdam. The two knobs studied in the paper:

- `precond_power` (the anisotropy dial `p`): the Adam denominator is
  `denom_i = s_i^p * sbar^(1 - p)` with `s_i = sqrt(vhat_i)` and `sbar` a shared scalar.
  `p=1` is standard per-coordinate Adam; `p=0` is exactly gauge-invariant only when
  `precond_scalar='rms'`.
- `precond_scalar`: `'rms'` (exactly gauge-invariant at `p=0`) or `'geomean'` (legacy, not
  exactly gauge-invariant at finite step size).

```python
import torch
from flowadam import FlowAdam

model = torch.nn.Linear(64, 64)
x = torch.randn(8, 64)
opt = FlowAdam(model.parameters(), lr=1e-3,
               precond_power=0.0, precond_scalar='rms',   # the gauge-invariant dial endpoint
               clip_mode='globalnorm', clip_norm_c=10.0)

def closure():
    opt.zero_grad()
    loss = model(x).pow(2).mean()
    loss.backward()
    return loss

opt.step(closure)   # FlowAdam requires a closure
```

An exactly gauge-invariant `p=0` run needs both `precond_scalar='rms'` and
`clip_mode='globalnorm'` — the default per-coordinate clip breaks the gauge on any step where the
ODE path triggers. The package default scalar is still `'geomean'` for backwards compatibility;
don't use it for an exact finite-step equivariance claim. FlowAdam takes one parameter group.

## Repository layout

```
flowadam/            the optimizer package
experiments/         CPU experiments (the mechanism-identification suite)
experiments/nibi/    the H100 replication ladder (GPU)
logs/                checked-in stdout and JSONL from the CPU runs
figures/             paper-figure generation
tests/               numerical verification of every identity in the paper
```

## Reproducing the paper

Run the CPU experiments from `experiments/` (sibling imports resolve there):

```bash
cd experiments
python optimizer_zoo_bias.py
```

| Script | Paper | What it shows |
| --- | --- | --- |
| `optimizer_zoo_bias.py` | Section 5 | Optimizer-zoo map: the equivariant/coordinate-wise split in recovery |
| `zoo_lr_sensitivity.py` | Section 5, App D.2 | Learning-rate curves for the zoo comparison |
| `zoo_decay_control.py` | App D.1 | Schedule symmetrization: split survives cosine-for-all |
| `zoo_init_scale.py` | App D.4 | Split persists across initialization scales |
| `zoo_size_check.py` | Section 5 | The ordering survives a second size and rank |
| `nuclear_norm_reference.py` | Section 5, Table 2 | Min-nuclear-norm interpolant on the same three instances: the reference row |
| `precond_dial_scalar_check.py` | Section 7, Figure 3 | The dial `p: 1 -> 0` under both scalar conventions; the envelope arm of Figure 3 |
| `precond_dial_fixed_lr.py` | Section 7, App C | The dial's fixed-step arm: the `p`-sweep repeated at each rate of the shared grid (output in `precond_dial_fixed_lr.log`) |
| `equivariance_balance_probe.py` | Section 7 | Gauge product-drift and the balancedness invariant |
| `muon_shampoo_bias_probe.py` | Section 5 | Muon and Shampoo preserve the bias (geometry, not balancedness) |
| `muon_phase_diagram.py` | Section 8 | Spectral-tail phase diagram: Muon's two regimes and the crossing |
| `phase_diagram_decay_control.py` | App D.1 | Schedule control for the phase diagram |
| `signum_c9_audit.py` | App D.10 | Recovery without equivariance: annealed sign descent |
| `restoration_probe.py` | (shared) | The matrix-sensing testbed used across the sensing experiments |
| `attention_gauge.py` | Section 6 | Attention-gauge onset: Adam splits after one step, SGD and scalar-Adam stay at numerical scale; Muon's gauge and noise twins can both diverge under numerical chaos |
| `attention_gauge_multiseed.py` | App D.5 | Attention gauge across seeds, gauge draws, and noise scales |
| `hyperspectral_completion.py` | Legacy smoke test | Indian Pines loader and exploratory smoke test; not a paper result (selection is test-informed) |
| `hyperspectral_wilson_v2.py` | Legacy protocol | Earlier fixed-learning-rate matched-loss protocol |
| `hyperspectral_wilson_v3.py` | Section 9, App D.6 | Canonical CPU Indian Pines reproduction: matched loss, train-only lr selection |
| `flowadam_upgrade.py` | Section 10 | The two clip modes: per-coordinate (0.347) and global-norm (0.220), plus the legacy geometric-mean dial |
| `flowadam_upgrade_rms.py` | Section 10 (canonical) | FlowAdam-p under the RMS scalar; the source of the reported 0.169 |
| `flowadam_p_interp_control.py` | App D.8 | Interpolation control (not an early-stopping artifact) |

Use `hyperspectral_wilson_v3.py` for the CPU real-data protocol. The two legacy scripts are kept
for loader coverage and protocol comparison only.

### The H100 replication ladder (Appendix D.9)

`experiments/nibi/` is a self-contained GPU suite, device- and dtype-agnostic ports of the CPU
protocols. Run the drivers, then aggregate:

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

Other ladder sizes go to separate files, e.g.
`python zoo_ladder.py --n 128 --out ../nibi_results/zoo_n128.jsonl`. For Indian Pines use
`--dataset indianpines --densities 0.15,0.25`.

### Figures

```bash
cd figures
python make_figures.py               # Figures 1, 2, 3, 4
python make_realdata_trajectory.py   # Figure 5 (reads experiments/nibi_results/indianpines_gpu.jsonl)
```

Both write vector PDFs into `figures/`; these are the figures the manuscript includes. Figure 5
is separate because it reads the committed GPU records rather than a local run.

Figures 1 and 5 are not transcribed. Figure 1 recomputes Table 2's learning-rate selection from
the per-(method, lr, seed) records in `logs/optimizer_zoo_bias.jsonl`. Figure 5 re-derives
Appendix D.6's train-only rule from the GPU records, selecting on seeds 42 and 123 with 456 and
789 held out. Both assert they land on the published numbers, so neither can drift from the raw
output without failing.

The attention-gauge curves are a snapshot of `logs/attention_gauge_cpu.txt` and
`logs/attention_gauge_noise_cpu.txt`. Regenerate the logs and the figure together whenever the
attention implementation changes.

## Raw records

`logs/` holds the CPU output behind Section 5, each run a few laptop-minutes to reproduce:

- `optimizer_zoo_bias.jsonl` — one record per (method, lr, seed) over the full grid, plus one
  `selected` record per method. `optimizer_zoo_bias.txt` is the same run's stdout. Together they
  cover every cell of Table 2 and every bar of Figure 1, selection included.
- `nuclear_norm_reference.txt` — Table 2's convex reference row.
- `zoo_lr_sensitivity.txt`, `zoo_decay_control.txt`, `zoo_init_scale.txt`, `zoo_size_check.txt` —
  the Appendix D controls that row rests on.

`experiments/nibi_results/` holds the H100 JSONL behind Table 5 (GPU hyperspectral and Pavia),
Table 9 (twin drift at scale) and Table 11 (the problem-size ladder). These runs need a GPU
allocation, so the records are committed to make the tables checkable directly. It is also the
directory the ladder writes to, so `python collect.py --dir ../nibi_results` regenerates
`SUMMARY.md` without a GPU.

Each `zoo_n*.jsonl` has one `select` record naming the learning rate chosen per method, plus one
record per (method, lr, seed). Table 11's cells are the mean `rec` over seeds at the selected rate.

`dial_n40_flowlong.jsonl` audits the one cell in `dial_n40.jsonl` that never interpolated:
FlowAdam-p at p=0, lr=1e-3, where all ten seeds hit the sweep's 30k-step cap above the 1e-7 bar.
`experiments/nibi/dial_flowlong.py` reruns those seeds at the same rate with a 300k budget. Every
one crosses the bar between 38k and 59k steps, and mean recovery comes back 0.1475 +- 0.0357
against 0.1476 early-stopped, so Section 10's 0.148 is an interpolating reading. Selection applies
the bar to whatever budget it was given, so re-collecting over `dial_n40.jsonl` alone still passes
that rate over; the extended file is what settles the cell.

## Data

The synthetic sensing tasks are generated in code. The real-data experiments use two public
hyperspectral scenes and a character-level text corpus, none of them shipped here:

- **Indian Pines** (`Indian_pines_corrected.mat`, key `indian_pines_corrected`) and
  **Pavia University** (`PaviaU.mat`, key `paviaU`), both from the hyperspectral remote-sensing
  scenes collection at the University of the Basque Country (ehu.eus). Put
  `Indian_pines_corrected.mat` under `data/hyperspectral/` for the CPU scripts, or pass `--mat`
  to `experiments/nibi/pavia.py`.
- **tiny-Shakespeare** for the character-LM attention twins (`input.txt`, ~1.1 MB, from
  `data/tinyshakespeare/input.txt` in github.com/karpathy/char-rnn). Put it at
  `data/text/input.txt` or pass `--data`. The paper's char-LM row uses
  `--task text --depth 6 --dmodel 256`, other flags at defaults.

## Citation

```bibtex
@article{singh2026gauge,
  title  = {The Loss Does Not See the Basis, but Adam Does},
  author = {Singh, Devender},
  year   = {2026},
  note   = {Preprint}
}
```

## License

MIT. See [LICENSE](LICENSE).
