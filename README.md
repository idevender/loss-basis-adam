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
figures/             paper-figure generation
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
| `zoo_lr_sensitivity.py` | Section 5.1, App C2 | Learning-rate curves for the optimizer-zoo comparison |
| `zoo_decay_control.py` | App C1 | Schedule symmetrization: split survives cosine-for-all |
| `zoo_init_scale.py` | App C3 | Split persists across initialization scales |
| `zoo_size_check.py` | Section 5 | The ordering survives a second size and rank |
| `precond_dial_scalar_check.py` | Section 6 | The anisotropy dial `p: 1 -> 0` under both scalar conventions |
| `equivariance_balance_probe.py` | Section 6 | Gauge product-drift and the balancedness invariant |
| `muon_shampoo_bias_probe.py` | Section 5 | Muon and Shampoo preserve the bias (geometry, not balancedness) |
| `muon_phase_diagram.py` | Section 8 | Spectral-tail phase diagram: Muon's two regimes and the crossing |
| `phase_diagram_decay_control.py` | App C1 | Schedule control for the phase diagram |
| `signum_c9_audit.py` | App C9 | Recovery without equivariance: annealed sign descent |
| `restoration_probe.py` | (shared) | The matrix-sensing testbed used across the sensing experiments |
| `attention_gauge.py` | Section 7 | Attention-gauge onset: Adam splits after one step; SGD/scalar-Adam stay at numerical scale, while Muon gauge and noise twins can both diverge under numerical chaos |
| `attention_gauge_multiseed.py` | App C4 | Attention gauge across seeds, gauge draws, and noise scales |
| `hyperspectral_completion.py` | Legacy smoke test | Indian Pines loader and exploratory CPU smoke test; configuration selection is test-informed and it is not a paper result |
| `hyperspectral_wilson_v2.py` | Legacy protocol | Earlier fixed-learning-rate matched-loss protocol, retained for comparison |
| `hyperspectral_wilson_v3.py` | Section 9, App C5 | Canonical CPU Indian Pines reproduction: matched loss with train-only learning-rate selection |
| `flowadam_upgrade.py` | Section 10 (legacy scalar) | FlowAdam-p under the legacy geometric-mean scalar; robustness comparison, not the reported numbers |
| `flowadam_upgrade_rms.py` | Section 10 (canonical) | FlowAdam-p under the paper's RMS scalar; the source of the reported Section 10 numbers |
| `flowadam_p_interp_control.py` | App C7 | Interpolation control (not an early-stopping artifact) |

For the paper's CPU real-data protocol, use `hyperspectral_wilson_v3.py`. The two legacy scripts
remain for loader coverage and protocol comparison; they are not the source of the paper's reported
real-data numbers.

### The H100 replication ladder (Appendix C8)

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
python make_figures.py
```

The script writes vector PDFs directly to `figures/`. These are the exact figures the manuscript
includes. Its attention-gauge curves are a checked
snapshot of the current CPU runs recorded in `logs/attention_gauge_cpu.txt` and
`logs/attention_gauge_noise_cpu.txt`; regenerate the logs and figure together whenever the attention
implementation changes.

## Raw cluster records

`experiments/nibi_results/` holds the raw JSONL records from the H100 runs behind Table 5
(hyperspectral GPU replication and Pavia), Table 9 (twin drift at scale) and Table 11 (the
problem-size ladder). They are committed because those runs need a GPU allocation to reproduce, so
the records let you check the tables directly. This is the same directory the ladder writes to above,
so `python collect.py --dir ../nibi_results` regenerates `SUMMARY.md` from them without a GPU.

Each `zoo_n*.jsonl` carries one `select` record naming the learning rate chosen for each method, plus
one record per (method, lr, seed). Table 11's cells are the mean `rec` over the seeds at each method's
selected rate.


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
