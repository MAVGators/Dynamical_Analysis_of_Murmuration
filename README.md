# Murmuration on My Mind

A physics project modeling the kinematics and information propagation of starling murmurations.

**Caiman Moreno-Earle** — California Institute of Technology (Ph 11 Hurdle)

The full writeup is [writeup.pdf](writeup.pdf) (source: [writeup.tex](writeup.tex)).

## Overview

The project attacks flocking behavior from two directions:

1. **Swarm on a Sphere (SOS)** — a continuous-time model where each bird's state is a unit
   heading vector on $S^2$. Birds align via a projected consensus term restricted to the
   tangent space of the sphere, so headings stay normalized. Simulations sweep flock size,
   number of tracked neighbors, interaction weight, and initial alignment spread to measure
   how fast a flock converges to a common heading. The writeup pairs this with a Lyapunov
   argument for global asymptotic stability of the flocked state.

2. **Heisenberg / Glauber dynamics** — the mean-field magnetization equation borrowed from
   statistical mechanics, used as a proxy for how fast information (e.g. a predator sighting)
   propagates across the flock. Adding a time-varying external field $B(t) = B_0\sin(\omega t)$
   models a predator attack, and sweeping $B_0$ and $\omega$ gives a response threshold.

Key results: information crosses the swarm in roughly 0.6 s, and the effective neighbor
count is $n \approx 6$.

## Files

| File | Purpose |
| --- | --- |
| [SOS.py](SOS.py) | Main SOS simulation: initial conditions, ODE integration, convergence metrics, 3D animations, parameter sweeps |
| [SOS_interactions.py](SOS_interactions.py) | Interaction-graph builders — fully connected, no interaction, and topological (lattice) neighbors |
| [lattice.py](lattice.py) | `Lattice` class; fixed 3D lattice of bird positions used for the short-time approximation |
| [HG.py](HG.py) | Heisenberg/Glauber magnetization model, including the driven predator-response case |
| [writeup.tex](writeup.tex) / [writeup.pdf](writeup.pdf) | Full LaTeX writeup |
| [refs.bib](refs.bib) | Bibliography (biblatex/biber) |
| [figures/](figures/) | Figures used in the writeup |
| [results/](results/) | Generated simulation output (convergence plots, GIF animations) |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
pip install scipy tqdm         # also required, not yet in requirements.txt
```

## Running

Both simulation scripts drive themselves from a `__main__` block, so you pick an
experiment by uncommenting the call you want.

**SOS model** — [SOS.py:668-673](SOS.py#L668-L673) selects between:

- `t1()` — single run on a 5×5×5 lattice; saves a convergence plot and a 3D animation
- `t2()` — full parameter sweep over lattice size, initial spread, neighbor order, and weight
- `two_swarm_test()` — two counter-aligned swarms merging
- `overlay_two_swarm_vs_random(...)` — two-swarm vs. fully random convergence comparison
- `visualize_initial_headings_3d(...)` — diagnostic plot of the initial lattice + headings

```bash
python SOS.py
```

Output is written to `results/` with the parameters encoded in the filename
(e.g. `sos_x-5_y-5_z-5_initial_radius-0.5_order-1_weight-1.png`).

**Heisenberg model** — [HG.py:105-118](HG.py#L105-L118) sets the parameters and calls one of
`sweep_omega_overlay` / `sweep_B0_overlay` / `plot_solution`. These display plots interactively
rather than saving them.

```bash
python HG.py
```

### Parameters worth knowing

**SOS** — `x, y, z` (lattice dimensions, so $N = xyz$ birds), `initial_radius` (angular spread
of initial headings; 0 = perfectly aligned, 1 = uniform on the sphere), `order` (neighbor
shell — 1 gives 6 neighbors in 3D), `weight` (coupling strength), `steps` and `t_span`
(integration grid).

**Heisenberg** — `alpha` (relaxation rate), `J` (coupling), `z` (coordination number),
`Tk` ($k_BT$; noise level), `B0` and `omega` (predator field amplitude and frequency).

## Building the writeup

```bash
latexmk -pdf writeup.tex
```

Requires biber for the bibliography.

## Notes

- `results/` paths in [SOS.py](SOS.py) are hardcoded with Windows backslashes, so saving
  output on Linux/macOS needs those changed.
- The `order`/`weight` combinations in the sweeps are paired deliberately: `order=2` uses
  `weight = 1/6` so total coupling stays comparable to `order=1, weight=1`.
- AI disclaimer, per the headers in [SOS.py](SOS.py) and [HG.py](HG.py): ChatGPT assisted with
  plotting/visualization code, `generate_save_path()`, and `sweep_parameters()`. All physics,
  equations, and model logic were written by the author.
