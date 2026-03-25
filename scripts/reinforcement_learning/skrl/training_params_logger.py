"""Training parameters logger — dumps training config to JSON for reproducibility."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def dump_training_params(log_dir: str, run_name: str, env_cfg, agent_cfg, args_cli) -> None:
    """Dump training parameters to a JSON file in log_dir."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    params = {
        "run_name": run_name,
        "args_cli": vars(args_cli) if hasattr(args_cli, "__dict__") else str(args_cli),
    }

    # Extract env_cfg fields safely
    try:
        params["env_cfg"] = str(env_cfg)
    except Exception:
        params["env_cfg"] = repr(env_cfg)

    # Extract agent_cfg fields safely
    try:
        if isinstance(agent_cfg, dict):
            params["agent_cfg"] = agent_cfg
        else:
            params["agent_cfg"] = str(agent_cfg)
    except Exception:
        params["agent_cfg"] = repr(agent_cfg)

    out_file = log_path / "training_params.json"
    try:
        with open(out_file, "w") as f:
            json.dump(params, f, indent=2, default=str)
        logger.info(f"Training params saved to {out_file}")
    except Exception as e:
        logger.warning(f"Failed to save training params: {e}")
