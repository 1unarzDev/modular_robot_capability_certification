# Modular robot capability certification

A team of docked modular robots can split into smaller robots or merge with another team, so it can reshape itself
for each job. Before it does, it needs to know **which of the configurations it could become can actually do the
job**. That is harder than it sounds. Modules built to one design are never quite alike: motors vary, batteries sag,
gearboxes wear and drivers degrade. Inside an assembly those differences stay hidden because neighbouring wheels share
the load. And a merge with another team creates a configuration that has never been driven.

This repository contains the code, data and analysis for **certified capability prediction**. From one 12-second
test drive of the current assembly, it gives a lower bound on how well every possible child or merged configuration
can move in each direction. It comes with a statistical guarantee on the chance of sending any configuration on a job
it cannot do.

## How it works

1. **Estimate each module's hidden state.** A learned estimator, a set transformer over wheels, reads the test drive
   and outputs a probability distribution over each wheel's condition and each module's motor and friction parameters.
2. **Build every candidate in simulation.** Each child and merge is assembled from those module estimates in a
   physics simulator and driven in ten motion directions.
3. **Add one calibrated safety margin.** A conformal margin, calibrated on whole episodes, turns the predictions into
   lower bounds that hold for every candidate at once. A planner can therefore safely pick the best certified one.

The analysis also shows which answers the test drive actually observed, and which ones rest on the motor model
(for example, diagonal moves and the turns of larger children). Those extrapolated answers are where an uncalibrated
physics model fails silently when the motor model is wrong.

## Results (simulation, pre-registered run)

- Only **0.95 %** of test episodes contained an unsafe commitment, against a 10 % limit.
- On faulty modules left alone after a split, the certificate certified **72.8 %** of capable decisions, against
  **51.8 %** for the strongest prior certificate, with fewer unsafe commitments.
- A compositional version certified **90.8 %** of capable merges of teams that had never been docked together.

The robot is a fleet of 24 cm mecanum-wheeled modules. The simulators are identified from logs of a real six-module
fleet, and a second, independently written simulator is used for stress tests. A paper describing the work is in
preparation.

## Repository layout

| Path | Contents |
|---|---|
| `cfcap/t0` | Quasi-static analysis model (wheel torque–speed limits, what a test drive can observe) |
| `cfcap/t1` | Batched dynamic simulator used for training, calibration and testing |
| `cfcap/t2` | Independently written stress simulator (dock compliance, load transfer, knee motor model) |
| `cfcap/gen` | Episode generator: parents, faults, motor variation, children and merges |
| `cfcap/models` | Learned estimator and composition of candidate configurations |
| `cfcap/conformal` | Episode-level conformal calibration and lower bounds |
| `cfcap/baselines` | Prior certificate and simpler baselines |
| `cfcap/decide`, `cfcap/probe` | Exploratory decision layer: when an extra test drive is worth its time |
| `scripts/` | Experiment, export and campaign scripts; `scripts/analysis/` regenerates every table, figure and number from `results/` |
| `results/` | Result CSVs, including the scored pre-registered run in `results/confirmatory/` |
| `docs/` | Study plan, hypotheses ledger, pre-registration (`CONFIRMATORY.md`) and its results |
| `tests/` | Unit and analysis tests |

## Running it

The dynamic simulator builds on the companion `capxfer` package, which is not yet public. `env.sh` sets the
interpreter and import paths; adjust it to your machine, then:

```bash
source env.sh
$PY -m pytest -q tests                          # tests
$PY scripts/analysis/confirmatory.py --mock /tmp/m   # the pre-registered scorer on mock data
```

The 128 MB model-results file `results/models/confirmatory/dec_conf_models.csv.gz` is stored with Git LFS.
