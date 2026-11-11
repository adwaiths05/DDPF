# DDPF-Net and Crowd Evacuation Reinforcement Learning

Dual-Decoder Point-supervised Framework for crowd counting and head
localization. The model uses a Swin-Tiny backbone and produces both a
density map and a localization heatmap.

The project is designed for the ShanghaiTech Crowd Counting Dataset
(Part B).

## Features

- Density-based crowd-count estimation.
- Head localization with confidence scores and point offsets.
- Training and validation metrics including MAE, RMSE, R2, Pearson
  correlation, precision, recall, and F1.
- Streamlit image-upload interface.
- JSON output for one image or a complete image folder.
- Diagnostic plots, checkpoints, and per-image evaluation reports.

## Project layout

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit web application for interactive inference. |
| `config.py` | Dataset paths, training settings, thresholds, and output paths. |
| `dataset.py` | ShanghaiTech image and ground-truth loading. |
| `engine.py` | Modular training and validation engine used by `train.py`. |
| `evaluation.py` | Validation and localization metrics. |
| `inference.py` | Image/folder inference implementation. |
| `inference_json.py` | Image/folder inference that saves JSON results. |
| `losses.py` | DDPF-Net loss functions. |
| `models.py` | Swin-Tiny DDPF-Net architecture. |
| `train.py` | Command-line entry point for the modular training pipeline. |
| `training.py` | Standalone training and evaluation pipeline with checkpoint resume and diagnostics. |
| `utils.py` | Density, heatmap, offset, and metric utilities. |

## Requirements

Use Python 3.10 or newer and install the required packages:

```bash
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision timm albumentations opencv-python numpy scipy scikit-learn matplotlib tqdm streamlit
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision timm albumentations opencv-python numpy scipy scikit-learn matplotlib tqdm streamlit
```

Install the PyTorch build appropriate for the available CUDA version if GPU
acceleration is required. CPU inference is supported, but training is
intended for a CUDA-capable machine.

## Dataset setup

Download and extract ShanghaiTech so the directory has this shape:

```text
data/
â””â”€â”€ part_B/
    â”œâ”€â”€ train_data/
    â”‚   â”œâ”€â”€ images/
    â”‚   â””â”€â”€ ground-truth/
    â””â”€â”€ test_data/
        â”œâ”€â”€ images/
        â””â”€â”€ ground-truth/
```

Update `ROOT`/`LOCAL_DATA_ROOT` and `DATASET_PART` in `config.py` to match
the local dataset. The default modular pipeline uses `part_B`.

Ground-truth files must use the ShanghaiTech naming convention expected by
`dataset.py`, for example `GT_IMG_1.mat`.

## Training

### Modular pipeline

Run the modular training and evaluation entry point:

```bash
python train.py
```

Select a fold with:

```bash
python train.py --fold 0
```

The modular pipeline writes checkpoints and metrics to the directory set by
`CHECKPOINT_DIR` in `config.py` (normally `fin_try/`).

### Standalone pipeline

`training.py` is a self-contained training/evaluation script. It also
supports automatic installation of a few missing Python packages:

```bash
python training.py
```

Start without resuming from the latest checkpoint:

```bash
python training.py --fresh-start
```

Evaluate the saved best checkpoint without training:

```bash
python training.py --eval-only
```

Its output directory and dataset settings are configured near the top of
`training.py`.

## JSON inference

`inference_json.py` accepts either one image or a folder and writes
per-image JSON files plus `summary.json`:

```bash
python inference_json.py `
  --input path\to\image-or-folder `
  --ckpt path\to\model.pth `
  --output-root inference_json_results
```

Optional detection controls:

```bash
python inference_json.py `
  --input path\to\images `
  --ckpt path\to\model.pth `
  --thresh 0.30 `
  --min_dist 4
```

Each JSON result contains:

- density-based count;
- localization-based count;
- overall and quadrant density scores;
- detected head coordinates and confidence values.

The script's default checkpoint search paths are defined in
`resolve_checkpoint_path`. Pass `--ckpt` explicitly when using a checkpoint
stored in a custom directory.

## Streamlit application

Start the interactive application from the repository root:

```bash
streamlit run app.py
```

Set `CHECKPOINT_PATH` in `app.py` to a compatible checkpoint before starting
the app. The application currently defaults to
`op/best_balanced_ddpfnet_final_try.pth`.

## Outputs

Training and diagnostics may generate:

- `.pth` model checkpoints;
- epoch metric CSV files;
- density/error/scatter plots;
- validation visualizations;
- failure-case images;
- JSON inference results.

These generated artifacts are intentionally excluded by `.gitignore`.
Source code, including `training.py` and `inference_json.py`, remains
trackable.

## Reproducibility

The default random seed is `42`. Results can still vary with GPU hardware,
CUDA/cuDNN versions, worker settings, and dataset preprocessing. Record the
dataset part, configuration values, checkpoint path, and detection threshold
when reporting results.

## License

This project is released under the MIT License. See `LICENSE`.

---

# Crowd Evacuation Reinforcement Learning

The commands in this section are run from the `rl/` directory. From the
repository root, enter it first:

```powershell
cd rl
```

This project is a research prototype for learning evacuation routing policies on synthetic indoor venue graphs. It contains:

- A graph-based crowd evacuation environment.
- PPO and hierarchical MAPPO reinforcement learning agents.
- Shortest-path and congestion-aware heuristic baselines.
- Reproducible venue and crowd generation.
- Checkpointing, demos, evaluation scripts, visualization helpers, and tests.

The environment models aggregate crowd movement. Each zone contains a number of occupants, and a policy chooses which outgoing corridor each zone should use at every step.

## Project Status

The current implementation includes the following completed work:

- PPO training and checkpoint loading.
- MAPPO training through a multi-agent environment wrapper.
- MAPPO checkpoint loading in the interactive demo.
- Correct action storage for PPO ratios instead of reconstructing actions with argmax.
- Action-mask storage and reuse during MAPPO updates.
- Generalized Advantage Estimation (GAE) with normalized advantages.
- Rollout-boundary value bootstrapping when an update ends mid-episode.
- A separate critic learning rate and Huber critic loss for better value stability.
- Averaged policy, value, and entropy metrics across update minibatches.
- Training from the environment's actual reward.
- Different random reset states between episodes.
- Timestamped MAPPO checkpoint and metric logs.
- MAPPO demo support through `scripts/run_demo.py`.

The MAPPO tests currently pass with four passing tests in `tests/test_mappo.py`.

## Repository Layout

```text
configs/
    env_config.yaml       Environment configuration
    eval_config.yaml      Evaluation configuration
    mappo.yaml            MAPPO configuration reference
    ppo.yaml              PPO configuration reference

scripts/
    evaluate.py           Compare baselines, PPO, and MAPPO
    run_demo.py           Run one interactive episode
    run_demo_test.py      Demo smoke-test helper
    train_mappo.py        MAPPO training entry point
    train_ppo.py          PPO training entry point
    visualize_evacuation.py Animate initial crowd and PPO/MAPPO evacuation

src/
    venue_generator.py    Synthetic graph and venue generation
    crowd_generator.py    Initial crowd and scenario generation
    evacuation_env.py     Gymnasium-compatible single-agent environment
    observation_encoder.py Fixed-size observation encoding
    reward_function.py    Decomposed reward calculation
    baselines.py          Shortest-path and congestion-aware policies
    evaluation.py         Monte Carlo policy evaluation
    visualization.py      Plotting and visualization helpers
    integration_adapter.py External integration interface
    ppo/                   PPO networks, buffer, agent, and trainer
    mappo/                 MAPPO roles, critic, wrapper, agent, and trainer

tests/
    Test coverage for environment, generators, observations, rewards, PPO, MAPPO, and integration.

results/
    ppo/                   PPO checkpoints
    mappo/                 MAPPO checkpoints
```

## Installation

Create or activate a Python environment, then install the dependencies:

```powershell
cd rl
pip install -r requirements.txt
```

The main dependencies are:

- `torch`
- `gymnasium`
- `numpy`
- `networkx`
- `matplotlib`
- `seaborn`
- `pandas`
- `pyyaml`
- `pytest`

For GPU training, install a PyTorch build compatible with the installed CUDA driver. The training code automatically falls back to CPU if `--device cuda` is requested but CUDA is unavailable.

Check CUDA availability with:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

## Quick Start

### Run a heuristic demo

```powershell
python scripts/run_demo.py --policy shortest_path
python scripts/run_demo.py --policy congestion_aware
```

### Run a PPO checkpoint

```powershell
python scripts/run_demo.py `
    --policy ppo `
    --checkpoint results/ppo/ppo_policy_final.pt
```

### Run a MAPPO checkpoint

```powershell
python scripts/run_demo.py `
    --policy mappo `
    --checkpoint results/mappo/mappo_policy_final.pt
```

The demo prints the generated venue, initial population, evacuation progress, reward, occupancy snapshots, and final outcome. Use `--max-steps` to shorten or extend an episode:

```powershell
python scripts/run_demo.py --policy mappo `
    --checkpoint results/mappo/mappo_policy_final.pt `
    --max-steps 100
```

## Visualize Evacuation

Use the visualization script to first display the initial crowd distribution, then animate deterministic PPO and MAPPO decisions side by side on the same generated venue:

```powershell
python scripts/visualize_evacuation.py
```

To save the animation as a GIF without opening a window:

```powershell
python scripts/visualize_evacuation.py `
    --max-steps 50 `
    --save results/evacuation.gif `
    --no-show
```

The animation contains two phases:

1. **Initial crowd view**: both panels show the same starting venue and the number of people in every zone.
2. **Evacuation comparison**: the left panel runs PPO and the right panel runs MAPPO using the same initial venue and seed.

Visual elements:

- Orange circles are zones. The label shows the zone ID and current occupancy.
- Green squares are exits.
- Orange edge colors and thicker edges indicate higher corridor congestion.
- The title reports the current step, evacuated population, and evacuation percentage.
- PPO and MAPPO use deterministic masked argmax actions for a stable visual comparison.

Useful options:

```powershell
python scripts/visualize_evacuation.py --seed 7 --max-steps 100
python scripts/visualize_evacuation.py `
    --ppo-checkpoint results/ppo/ppo_policy_final.pt `
    --mappo-checkpoint results/mappo/mappo_policy_final.pt `
    --save results/evacuation_seed7.gif `
    --seed 7 `
    --no-show
```

The script uses CPU inference so it can be used for demonstrations even when a GPU is unavailable. On Windows, it includes a compatibility setting for duplicate Intel OpenMP runtimes that can otherwise occur when PyTorch, NumPy, and Matplotlib are loaded together.

## Training PPO

Run the PPO trainer with its defaults:

```powershell
python scripts/train_ppo.py
```

Run a longer GPU training job:

```powershell
python scripts/train_ppo.py `
    --seed 42 `
    --max-steps 100000 `
    --device cuda
```

PPO checkpoints are written to `results/ppo` by default. The final policy is normally saved as:

```text
results/ppo/ppo_policy_final.pt
```

## Training MAPPO

Run MAPPO on CPU:

```powershell
python scripts/train_mappo.py `
    --seed 42 `
    --max-steps 100000 `
    --eval-interval 5000 `
    --device cpu
```

Run MAPPO on CUDA:

```powershell
python scripts/train_mappo.py `
    --seed 42 `
    --max-steps 100000 `
    --eval-interval 5000 `
    --device cuda
```

Arguments:

| Argument | Meaning | Default |
|---|---|---:|
| `--seed` | Reproducibility seed for venue and initial state | `42` |
| `--max-steps` | Total environment steps collected for training | `10000` |
| `--eval-interval` | Number of steps between MAPPO updates and checkpoints | `1000` |
| `--log-dir` | Directory for checkpoints | `results/mappo` |
| `--device` | `cpu` or `cuda`; falls back to CPU when CUDA is unavailable | `cpu` |

Each periodic log includes a local timestamp:

```text
[2026-10-27 14:32:10] [MAPPO] Step 5000/100000 - reward 0.35 - checkpoint results\\mappo\\mappo_step_5000.pt
[2026-10-27 14:32:10] Metrics: {'policy_loss': ..., 'value_loss': ..., 'entropy': ...}
```

The final checkpoint is saved as:

```text
results/mappo/mappo_policy_final.pt
```

Periodic checkpoints include the leader policy, zone policy, critic, optimizer state, and training step. The final checkpoint includes the leader policy, zone policy, and critic.

## Environment Model

### Venue graph

`VenueGraph` wraps a directed NetworkX graph. Nodes are either ordinary zones or exits. Edges represent corridors and contain travel time, capacity, blocked status, and congestion state.

The generator supports reproducible synthetic layouts, including grid, random, and multi-wing-style venues. Generated venues are validated so zones can reach at least one exit under normal conditions.

### Crowd state

The environment tracks:

- Occupants in every zone.
- Total occupants evacuated through exits.
- Corridor congestion.
- Blocked corridors.
- Unavailable exits.
- Hazard zones and risk values.
- Current simulation step.

Population is conserved according to:

```text
people remaining in zones + people evacuated = initial population
```

### Action space

The single-agent environment uses:

```text
MultiDiscrete([max_neighbors] * max_zones)
```

Each action entry selects an outgoing neighbor for one zone. Invalid actions are masked before policy sampling. Blocked corridors and unavailable exits are excluded from valid choices.

### Observation space

The observation is a fixed-size vector so that venues with different numbers of zones can use the same neural network dimensions. With `max_zones=15` and `max_neighbors=6`, the layout is:

| Section | Size | Description |
|---|---:|---|
| Per-zone features | `15 x 8` | Occupancy, capacity, risk, hazard, exit indicator, distance, neighbor count, normalized occupancy |
| Neighbor features | `15 x 6 x 2` | Neighbor occupancy and edge congestion |
| Global features | `5` | Remaining population, evacuated fraction, normalized time, blocked count, unavailable-exit count |
| Total | `185` | Fixed-size encoded observation |

The MAPPO wrapper slices the global observation into per-zone observations. A zone observation contains its eight local features plus its neighbor features, which is 20 values with six maximum neighbors.

### Rewards

The environment reward combines:

- Positive evacuation progress.
- Congestion penalty.
- Hazard exposure penalty.
- Invalid-action penalty.
- Time penalty.
- Completion bonus when everyone is evacuated.

MAPPO trains on the reward returned by the environment so that its training objective matches the reward shown during evaluation and demo execution.

## PPO

PPO is the single-agent baseline. It uses an actor network to produce logits for every zone action and a critic network for state-value estimation. The actor output is reshaped to:

```text
max_zones x max_neighbors
```

The action mask is applied before sampling or selecting actions.

Relevant files:

- `src/ppo/actor_critic.py`
- `src/ppo/rollout_buffer.py`
- `src/ppo/ppo_agent.py`
- `src/ppo/trainer.py`

## MAPPO

MAPPO uses the `MultiAgentEnv` wrapper and treats each zone as an agent. The wrapper converts a dictionary of zone actions into the full action vector expected by `EvacuationEnv`.

### Current MAPPO components

| Component | Current implementation |
|---|---|
| Zone policy | Local zone and neighbor observation to neighbor-selection logits |
| Centralized critic | Full concatenated per-zone observation to one state value |
| Leader policy | Defined and sampled, but not currently executed by the environment |
| Exit policy | Defined as a placeholder, not currently executed |
| Signage policy | Defined as a placeholder, not currently executed |
| Action masking | Stored during rollout and applied during the PPO update |
| Advantage estimation | GAE with normalized advantages |
| Critic update | Huber loss with a separate critic learning rate |

### MAPPO training improvements

The MAPPO update includes several stability improvements:

1. The PPO ratio uses the actual action sampled during rollout. It does not reconstruct the action with argmax.
2. Zone action masks are retained for every transition and reused during the update.
3. GAE reduces the variance of policy advantages.
4. Advantages are normalized before the actor update.
5. Incomplete rollouts bootstrap from the critic value of the next state.
6. The critic uses Huber loss to reduce the effect of large return outliers.
7. The critic has its own learning rate, currently `1e-3`, while actor parameters use `3e-4`.
8. Reported losses are averaged over update minibatches.
9. The current environment ignores leader actions, so the leader policy loss is excluded from the effective training objective. This prevents a non-causal policy from adding gradient noise to the zone policy.

## Evaluation

Evaluate heuristic baselines:

```powershell
python scripts/evaluate.py --n-episodes 20
```

Evaluate a trained PPO policy:

```powershell
python scripts/evaluate.py `
    --ppo-ckpt results/ppo/ppo_policy_final.pt `
    --n-episodes 20 `
    --output results/ppo_evaluation.json
```

Evaluate a trained MAPPO policy:

```powershell
python scripts/evaluate.py `
    --mappo-ckpt results/mappo/mappo_policy_final.pt `
    --n-episodes 20 `
    --output results/mappo_evaluation.json
```

Compare all available policies:

```powershell
python scripts/evaluate.py `
    --all `
    --n-episodes 20 `
    --output results/evaluation.json
```

Evaluation reports average reward and success rate. A successful episode is one where all occupants reach exits before the environment timeout.

For a meaningful comparison, use the same episode count and seed range for every policy. A two-step demo is only a smoke test; it is not enough to judge evacuation performance.

## Recorded Evaluation Results

The following results were generated with:

```powershell
python scripts/evaluate.py --all --n-episodes 20 --output results/eval_results.json
```

The evaluation used 20 episodes and the same evaluation seed range for every policy. A success means that all occupants were evacuated before the episode timeout.

| Policy | Average reward | Reward standard deviation | Success rate |
|---|---:|---:|---:|
| Shortest path | `-43.323` | `108.631` | `5%` |
| Congestion aware | `53.226` | `19.388` | `100%` |
| PPO | `267.779` | `194.453` | `50%` |
| MAPPO | `415.824` | `188.232` | `70%` |

### Interpretation

- The congestion-aware heuristic achieved the highest reliability, succeeding in every episode.
- MAPPO achieved the highest average reward among the learned policies and succeeded in 14 of 20 episodes.
- PPO succeeded in 10 of 20 episodes.
- The shortest-path heuristic had a low success rate and high reward variability. Shortest path alone does not account sufficiently for corridor capacity and congestion.
- MAPPO's average reward was approximately `148.045` points higher than PPO's average reward in this run.
- The learned policies have substantial reward variation across venues, so success rate and reward standard deviation should be considered together with the average.

### Per-episode rewards

The complete `episode_rewards` values stored in `results/eval_results.json` were:

```text
Shortest path:
-79.273, -75.613, -79.211, -73.132, -80.284,
-71.728, -81.529, -76.965, -69.310, -73.607,
82.062, -72.491, -70.811, -75.439, -73.869,
-79.243, -71.781, -79.837, 405.785, -70.188

Congestion aware:
105.067, 40.558, 50.014, 43.959, 45.813,
31.674, 87.613, 46.723, 52.384, 43.370,
84.744, 45.661, 40.252, 63.372, 37.296,
43.754, 44.028, 44.734, 77.785, 35.717

PPO:
22.531, 297.565, 232.940, 642.118, 471.422,
73.459, 356.433, 365.652, 95.533, 277.799,
72.320, 109.869, 208.584, 260.031, 3.433,
725.541, 241.337, 329.945, 89.385, 479.692

MAPPO:
573.766, 205.385, 588.481, 494.470, 408.053,
412.871, 220.334, 269.532, 732.609, 518.682,
60.224, 487.087, 502.052, 567.805, 632.194,
694.890, 298.589, 306.535, 192.782, 150.143
```

The JSON file remains the machine-readable source of truth. Rerun the evaluation command after training a new checkpoint before comparing results.

## Visualizing Training Curves

The project has plotting functions in `src/visualization.py`, including `plot_training_curve`. That function expects a CSV file with a `step` column and one metric column, for example `reward` or `value_loss`:

```python
from src.visualization import plot_training_curve

plot_training_curve(
    "results/mappo_training.csv",
    metric="reward",
    save_path="results/mappo_reward_curve.png",
)
```

At present, MAPPO prints timestamped metrics to the terminal but does not automatically write a CSV training log. The `.pt` checkpoints contain model weights and optimizer state, not the historical metric series, so a training curve cannot be reconstructed from the existing checkpoint files alone.

The current timestamped output is useful for manual logging. For persistent curves, redirect the training output to a text log and convert the periodic `Step` and `Metrics` lines to CSV, or add CSV logging to `src/mappo/trainer.py`. The expected CSV format is:

```text
step,reward,policy_loss,value_loss,entropy
5000,0.35,1.02,12.4,1.93
10000,3.48,0.84,10.7,1.98
```

Then plot individual metrics with:

```python
from src.visualization import plot_training_curve

plot_training_curve("results/mappo_training.csv", "reward", "results/mappo_reward.png")
plot_training_curve("results/mappo_training.csv", "value_loss", "results/mappo_value_loss.png")
plot_training_curve("results/mappo_training.csv", "entropy", "results/mappo_entropy.png")
```

The visualization module also provides `plot_venue` for venue layouts and `plot_episode_trajectory` for position arrays when those data are available.

## Testing

Run the complete test suite:

```powershell
python -m pytest tests/ -v --tb=short
```

Run the focused MAPPO tests:

```powershell
python -m pytest tests/test_mappo.py -q
```

The MAPPO tests cover:

- Multi-agent environment reset behavior.
- Multi-agent action stepping.
- MAPPO action selection and action ranges.
- A short MAPPO update smoke test.

## Checkpoints and Results

Typical output files are:

```text
results/ppo/ppo_step_1000.pt
results/ppo/ppo_policy_final.pt
results/mappo/mappo_step_5000.pt
results/mappo/mappo_policy_final.pt
results/evaluation.json
```

Checkpoint dimensions must match the environment configuration used during training. The current default architecture expects `max_zones=15` and `max_neighbors=6`.

## Integration Interface

The integration adapter provides a small interface for external callers:

```python
from src.integration_adapter import IntegrationAdapter

adapter = IntegrationAdapter(
    policy_type="ppo",
    checkpoint_path="results/ppo/ppo_policy_final.pt",
)

observation = adapter.initialize(seed=42)
result = adapter.step()
observation = adapter.reset(seed=43)
adapter.close()
```

The adapter is intended as a boundary for connecting the prototype to another application or data pipeline.

## Known Limitations

- Venue layouts and crowd behavior are synthetic.
- Crowd movement is aggregate flow, not individual pedestrian simulation.
- The environment has fixed maximum zone and neighbor dimensions.
- The leader policy is currently sampled but does not control zone actions.
- Exit and signage policies are placeholders and are not currently connected to environment actions.
- MAPPO evaluation uses the zone policy for routing and strips the unused leader action.
- The current evaluation is Monte Carlo based and does not provide confidence intervals.
- Training logs are printed to the terminal; there is no TensorBoard integration yet.
- Real venue data and external crowd prediction models are not connected to the environment.

## Recommended Next Steps

1. Connect the leader action to a meaningful zone-policy conditioning mechanism.
2. Add real exit and signage actions to the environment before training those policies.
3. Add average episode reward, evacuation percentage, success rate, and explained variance to training logs.
4. Add TensorBoard or CSV logging for long training runs.
5. Evaluate multiple seeds and compare MAPPO against PPO and heuristic baselines.
6. Add checkpoint resume support for interrupted long runs.
