import sys
# Ensure the repository root is on the Python path so that ``src`` is importable.
repo_root = "d:/Projects/final_year/Implementation/RL"
if repo_root not in sys.path:
    sys.path.append(repo_root)

# Import modules via the ``src`` package.
from src.venue_generator import VenueGenerator
from src.evacuation_env import EvacuationEnv
from src.crowd_generator import generate_crowd_scenario, ScenarioType

def _create_valid_venue(seed: int):
    """Try to create a random venue; if it fails validation, fall back to a grid.

    ``VenueGenerator.generate_random_venue`` occasionally produces a graph where
    some zones cannot reach any exit, which raises ``ValueError``. For a quick
    sanity‑check demo we prefer a guaranteed‑connected layout, so we catch the
    exception and retry with ``generate_grid_venue`` which always connects zones
    to exits.
    """
    try:
        return VenueGenerator.generate_random_venue(seed=seed)
    except ValueError as e:
        print("[run_demo_test] Random venue invalid – falling back to grid:", e)
        return VenueGenerator.generate_grid_venue(seed=seed)

if __name__ == "__main__":
    # Create a reproducible venue and crowd scenario.
    vg = _create_valid_venue(seed=123)
    cs = generate_crowd_scenario(vg, scenario_type=ScenarioType.MIXED, seed=456)
    # Note: the current EvacuationEnv does not ingest the crowd scenario object.

    env = EvacuationEnv(vg, seed=789)
    obs, info = env.reset(seed=100)
    print("Initial observation shape:", obs.shape)
    for step_idx in range(5):
        masks = env.action_masks()
        actions = []
        for i in range(env.max_zones):
            valid = [j for j, v in enumerate(masks[i]) if v]
            actions.append(valid[0] if valid else 0)
        obs, reward, done, trunc, info = env.step(actions)
        print(f"Step {step_idx+1} – reward {reward:.2f}, evacuated {info['evacuated']}")
        if done:
            break
