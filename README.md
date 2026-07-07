# The Loss Does Not See the Basis, but Adam Does

Code for the paper **"The Loss Does Not See the Basis, but Adam Does: Gauge Equivariance
Decides the Implicit Bias on Factored Models"** (Devender Singh).

A factored loss `L(UV^T)` is invariant under the gauge action `(U, V) -> (UQ, VQ)` for
orthogonal `Q`. This repository contains the experiments showing that an optimizer's implicit
bias on such models is decided by whether its update is *equivariant* to this action:
gauge-equivariant methods (gradient descent, momentum, shared-scalar Adam, Muon, Shampoo)
preserve gradient descent's low-rank bias, while coordinate-wise methods (Adam, RMSProp, Lion,
signum, Adafactor) read the arbitrary basis and destroy it. The same symmetry lives in every
attention head, and the mechanism is used constructively to repair our own optimizer, FlowAdam.

## Installation

```bash
git clone https://github.com/idevender/loss-basis-adam.git
cd loss-basis-adam
pip install -e .        # installs the flowadam package
pip install -r requirements.txt
```

Requires Python 3.9+. The CPU experiments run in minutes; the H100 replication ladder
(`experiments/nibi/`) needs a GPU.

## The optimizer

`flowadam/optimizer.py` implements FlowAdam. The two knobs studied in the paper are:

- `precond_power` (the anisotropy dial `p`): the Adam denominator is
  `denom_i = s_i^p * sbar^(1 - p)` with `s_i = sqrt(vhat_i)` and `sbar` a shared scalar.
  `p=1` is standard per-coordinate Adam; `p=0` is a gauge-invariant shared-scalar denominator.
- `precond_scalar`: the shared-scalar convention, `'rms'` (exactly gauge-invariant at `p=0`)
  or `'geomean'` (the legacy convention).

```python
import torch
from flowadam import FlowAdam

model = torch.nn.Linear(64, 64)
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
| `zoo_lr_sensitivity.py` | Section 5.1, App C2 | The split holds at every learning rate (flow-limit reading) |
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
| `attention_gauge.py` | Section 7 | The gauge in an attention head: Adam's twins split, equivariant twins do not |
| `attention_gauge_multiseed.py` | App C4 | Attention gauge across seeds, gauge draws, and noise scales |
| `hyperspectral_completion.py` | Section 9 | Indian Pines loaders and CPU smoke test |
| `hyperspectral_wilson_v2.py` | Section 9 | Matched-train-loss protocol on real data |
| `hyperspectral_wilson_v3.py` | Section 9, App C5 | Matched-loss with train-only learning-rate selection |
| `flowadam_upgrade.py` | Section 10 | FlowAdam-p: the flow plus the softened preconditioner |
| `flowadam_upgrade_rms.py` | Section 10 | FlowAdam-p numbers under the RMS scalar |
| `flowadam_p_interp_control.py` | App C7 | Interpolation control (not an early-stopping artifact) |

### The H100 replication ladder (Appendix C8)

`experiments/nibi/` is a self-contained GPU suite (device- and dtype-agnostic ports of the CPU
protocols). Run the drivers, then aggregate:

```bash
cd experiments/nibi
python zoo_ladder.py            # optimizer-zoo ladder across sizes/ranks (the C8 table)
python phase_fine.py           # spectral-tail phase diagram at scale
python dial_scale.py           # the anisotropy dial and FlowAdam-p at scale
python attention_suite.py --task mod
python pavia.py --dataset paviau --mat /path/to/PaviaU.mat
python collect.py --dir ../nibi_results --md ../nibi_results/SUMMARY.md
```

### Figures

```bash
cd figures
python make_figures.py
```

The plotted numbers are the final experiment outputs, embedded in the script.

## Data

The synthetic sensing tasks are generated in code. The real-data experiments use two public
hyperspectral scenes and a character-level text corpus, none of which are shipped here:

- **Indian Pines** (`Indian_pines_corrected.mat`, key `indian_pines_corrected`)
- **Pavia University** (`PaviaU.mat`, key `paviaU`)

  Both are available from the hyperspectral remote-sensing scenes collection at the
  University of the Basque Country (ehu.eus). Place `Indian_pines_corrected.mat` under
  `data/hyperspectral/` for the CPU scripts, or pass `--mat` to `experiments/nibi/pavia.py`.

- A plain-text corpus (e.g. tiny-shakespeare) for the character-LM attention twins in
  `experiments/nibi/attention_suite.py --task text`.

## Citation

```bibtex
@article{singh2026gauge,
  title  = {The Loss Does Not See the Basis, but Adam Does: Gauge Equivariance
            Decides the Implicit Bias on Factored Models},
  author = {Singh, Devender},
  year   = {2026},
  note   = {Preprint}
}
```

## License

MIT. See [LICENSE](LICENSE).
