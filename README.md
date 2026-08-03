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

Requires Python 3.9+. Individual CPU experiments run in minutes; the complete suite takes longer
and depends on hardware. The replication ladder (`experiments/nibi/`) is intended for a GPU.

## Every identity in the paper is a test

`tests/test_paper_identities.py` numerically verifies essentially every constructive mathematical
statement in the paper. Each test names the paper object it checks and evaluates *both* sides of
the claimed identity in float64 at random points — from a real loss, real autograd gradients, and
the real update rules (Muon's Newton–Schulz iteration and Shampoo's inverse roots are imported from
the experiment code that produced the tables) — then asserts agreement at machine precision. Every
positive claim is paired with a negative control that must *fail* the same check, so a harness bug
that trivially equated the two runs would not pass silently.

The suite covers: gradient covariance by autograd; multi-step equivariance of all four equivariant
rules; the step-1 gauge break for all five coordinate-wise rules; the exact witness matrices printed
in the proofs (the sign witness, the Adafactor matrices, the scalar one-step products); the
Gram-determined structure theorem; the spectral transfer function; one-step undamped Shampoo =
`msign`; Newton–Schulz equivariance at every truncation; the transfer theorem's time change;
balancedness conservation and its anisotropic drift; and both halves of the greedy/equal-rate
boundary proposition, including non-vacuousness of its tail bound.

```bash
pip install -r requirements.txt
pytest -q tests/test_paper_identities.py      # 25 tests, 41 parametrized runs
```

Test names carry the current manuscript's numbering (for example
`test_thm46_transfer_theorem_time_change_direction` for the transfer theorem, Theorem 4.6).

## The optimizer

`flowadam/optimizer.py` implements FlowAdam. The two knobs studied in the paper are:

- `precond_power` (the anisotropy dial `p`): the Adam denominator is
  `denom_i = s_i^p * sbar^(1 - p)` with `s_i = sqrt(vhat_i)` and `sbar` a shared scalar.
  `p=1` is standard per-coordinate Adam; `p=0` is an exactly gauge-invariant shared-scalar
  denominator only when `precond_scalar='rms'`.
- `precond_scalar`: the shared-scalar convention, `'rms'` (exactly gauge-invariant at `p=0`)
  or `'geomean'` (a legacy convention that is not exactly gauge-invariant at finite step size).

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

For an exactly gauge-invariant `p=0` run, pass both `precond_scalar='rms'` and
`clip_mode='globalnorm'`: the default per-coordinate clip is itself a gauge-breaking map on any
step where the ODE path triggers. The package default scalar remains `'geomean'` for backwards
compatibility; it should not be used for an exact finite-step equivariance claim. FlowAdam
currently accepts one parameter group.

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

Run the CPU experiments from the `experiments/` directory (sibling imports resolve there):

```bash
cd experiments
python optimizer_zoo_bias.py
```

| Script | Paper | What it shows |
| --- | --- | --- |
| `optimizer_zoo_bias.py` | Section 5 | Optimizer-zoo map: the equivariant/coordinate-wise split in recovery |
| `zoo_lr_sensitivity.py` | Section 5, App D.2 | Learning-rate curves for the optimizer-zoo comparison |
| `zoo_decay_control.py` | App D.1 | Schedule symmetrization: split survives cosine-for-all |
| `zoo_init_scale.py` | App D.4 | Split persists across initialization scales |
| `zoo_size_check.py` | Section 5 | The ordering survives a second size and rank |
| `nuclear_norm_reference.py` | Section 5, Table 2 | Min-nuclear-norm interpolant on the same three instances: the reference row |
| `precond_dial_scalar_check.py` | Section 7, Figure 3 | The anisotropy dial `p: 1 -> 0` under both scalar conventions; the envelope arm plotted in Figure 3 |
| `precond_dial_fixed_lr.py` | Section 7, App C | The dial's fixed-step arm: the full `p`-sweep repeated at each single rate of the shared grid (checked-in output in `precond_dial_fixed_lr.log`) |
| `equivariance_balance_probe.py` | Section 7 | Gauge product-drift and the balancedness invariant |
| `muon_shampoo_bias_probe.py` | Section 5 | Muon and Shampoo preserve the bias (geometry, not balancedness) |
| `muon_phase_diagram.py` | Section 8 | Spectral-tail phase diagram: Muon's two regimes and the crossing |
| `phase_diagram_decay_control.py` | App D.1 | Schedule control for the phase diagram |
| `signum_c9_audit.py` | App D.10 | Recovery without equivariance: annealed sign descent |
| `restoration_probe.py` | (shared) | The matrix-sensing testbed used across the sensing experiments |
| `attention_gauge.py` | Section 6 | Attention-gauge onset: Adam splits after one step; SGD/scalar-Adam stay at numerical scale, while Muon gauge and noise twins can both diverge under numerical chaos |
| `attention_gauge_multiseed.py` | App D.5 | Attention gauge across seeds, gauge draws, and noise scales |
| `hyperspectral_completion.py` | Legacy smoke test | Indian Pines loader and exploratory CPU smoke test; configuration selection is test-informed and it is not a paper result |
| `hyperspectral_wilson_v2.py` | Legacy protocol | Earlier fixed-learning-rate matched-loss protocol, retained for comparison |
| `hyperspectral_wilson_v3.py` | Section 9, App D.6 | Canonical CPU Indian Pines reproduction: matched loss with train-only learning-rate selection |
| `flowadam_upgrade.py` | Section 10 | The two clip modes: per-coordinate (0.347) and global-norm (0.220), plus the legacy geometric-mean dial |
| `flowadam_upgrade_rms.py` | Section 10 (canonical) | FlowAdam-p under the paper's RMS scalar; the source of the reported FlowAdam-p numbers (0.169) |
| `flowadam_p_interp_control.py` | App D.8 | Interpolation control (not an early-stopping artifact) |

For the paper's CPU real-data protocol, use `hyperspectral_wilson_v3.py`. The two legacy scripts
remain for loader coverage and protocol comparison; they are not the source of the paper's reported
real-data numbers.

### The H100 replication ladder (Appendix D.9)

`experiments/nibi/` is a self-contained GPU suite (device- and dtype-agnostic ports of the CPU
protocols). Run the drivers, then aggregate:

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

Run additional ladder sizes with separate files, for example
`python zoo_ladder.py --n 128 --out ../nibi_results/zoo_n128.jsonl`. For Indian Pines, use
`--dataset indianpines` and the paper densities `--densities 0.15,0.25`.

### Figures

```bash
cd figures
python make_figures.py            # Figures 1, 2, 3, 4
python make_realdata_trajectory.py   # Figure 5 (reads experiments/nibi_results/indianpines_gpu.jsonl)
```

The scripts write vector PDFs directly to `figures/`. These are the exact figures the manuscript
includes. Figure 5 has its own script because it reads the committed GPU records rather than a
local run.

Figures 1 and 5 are not transcribed. Figure 1 recomputes Table 2's learning-rate selection from
the per-(method, lr, seed) records in `logs/optimizer_zoo_bias.jsonl`; Figure 5 re-derives
Appendix D.6's train-only rule from the GPU records, selecting on seeds 42 and 123 with 456 and
789 held out. Both assert they land on the published numbers, so neither can drift from the raw
output without failing.

The attention-gauge curves are a checked snapshot of `logs/attention_gauge_cpu.txt` and
`logs/attention_gauge_noise_cpu.txt`; regenerate the logs and the figure together whenever the
attention implementation changes.

## Raw records

`logs/` holds the checked-in output of the CPU runs behind Section 5, each a few laptop-minutes to
reproduce. `optimizer_zoo_bias.jsonl` carries one record per (method, lr, seed) over every grid the
zoo sweeps plus one `selected` record per method, with the same run's stdout in
`optimizer_zoo_bias.txt`; between them every cell of Table 2 and every bar of Figure 1 checks
against raw output, selection included. `nuclear_norm_reference.txt` is Table 2's convex reference
row, and `zoo_lr_sensitivity.txt`, `zoo_decay_control.txt`, `zoo_init_scale.txt` and
`zoo_size_check.txt` the Appendix D controls it rests on.

`experiments/nibi_results/` holds the raw JSONL records from the H100 runs behind Table 5
(hyperspectral GPU replication and Pavia), Table 9 (twin drift at scale) and Table 11 (the
problem-size ladder). They are committed because those runs need a GPU allocation to reproduce, so
the records let you check the tables directly. This is the same directory the ladder writes to above,
so `python collect.py --dir ../nibi_results` regenerates `SUMMARY.md` from them without a GPU.

Each `zoo_n*.jsonl` carries one `select` record naming the learning rate chosen for each method, plus
one record per (method, lr, seed). Table 11's cells are the mean `rec` over the seeds at each method's
selected rate.

`dial_n40_flowlong.jsonl` audits the one cell in `dial_n40.jsonl` that never interpolated:
FlowAdam-p at p=0, lr=1e-3, where all ten seeds hit the sweep's 30k-step cap above the 1e-7 bar.
`experiments/nibi/dial_flowlong.py` reruns those seeds at the same rate with the budget raised to
300k; every one crosses the bar between 38k and 59k steps, and mean recovery comes back
0.1475 +- 0.0357 against 0.1476 early-stopped, so Section 10's 0.148 is an interpolating reading.
Selection applies the bar to the budget it was given, so re-collecting over `dial_n40.jsonl` alone
still passes that rate over; the extended file is what settles the cell.


## Data

The synthetic sensing tasks are generated in code. The real-data experiments use two public
hyperspectral scenes and a character-level text corpus, none of which are shipped here:

- **Indian Pines** (`Indian_pines_corrected.mat`, key `indian_pines_corrected`)
- **Pavia University** (`PaviaU.mat`, key `paviaU`)

  Both are available from the hyperspectral remote-sensing scenes collection at the
  University of the Basque Country (ehu.eus). Place `Indian_pines_corrected.mat` under
  `data/hyperspectral/` for the CPU scripts, or pass `--mat` to `experiments/nibi/pavia.py`.

- The tiny-Shakespeare corpus for the character-LM attention twins (`input.txt`, ~1.1 MB, from
  `data/tinyshakespeare/input.txt` in github.com/karpathy/char-rnn). Place it at
  `data/text/input.txt` or pass `--data`. The paper's char-LM row uses
  `--task text --depth 6 --dmodel 256` with the remaining flags at their defaults.

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
