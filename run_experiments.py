#!/usr/bin/env python3
"""
Script to run multiple experiments with different configurations.
"""

import argparse
import yaml
import subprocess
import sys
from pathlib import Path
from itertools import product


def generate_experiment_configs(base_config_path: str, experiment_grid: dict) -> list:
    """
    Generate experiment configurations from a grid.
    
    Args:
        base_config_path: Path to base config file
        experiment_grid: Dictionary of parameters to vary
    
    Returns:
        List of config dictionaries
    """
    # Load base config
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)
    
    # Generate all combinations
    keys = list(experiment_grid.keys())
    values = list(experiment_grid.values())
    
    configs = []
    for combination in product(*values):
        config = base_config.copy()
        for key, value in zip(keys, combination):
            # Set nested config values
            keys_path = key.split(".")
            d = config
            for k in keys_path[:-1]:
                if k not in d:
                    d[k] = {}
                d = d[k]
            d[keys_path[-1]] = value
        
        configs.append(config)
    
    return configs


def main():
    parser = argparse.ArgumentParser(description="Run multiple experiments")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to base config file",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default="experiments.yaml",
        help="Path to experiments configuration file",
    )
    args = parser.parse_args()
    
    # Load experiments config
    with open(args.experiments, "r") as f:
        experiments_config = yaml.safe_load(f)
    
    experiment_grid = experiments_config.get("grid", {})
    base_config_path = args.config
    
    # Generate configs
    configs = generate_experiment_configs(base_config_path, experiment_grid)
    
    print(f"Running {len(configs)} experiments...")
    
    # Run each experiment
    for i, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Experiment {i+1}/{len(configs)}")
        print(f"{'='*60}")
        
        # Save config to temporary file
        temp_config_path = f"temp_config_{i}.yaml"
        with open(temp_config_path, "w") as f:
            yaml.dump(config, f)
        
        try:
            # Run training
            result = subprocess.run(
                [sys.executable, "train.py", "--config", temp_config_path],
                check=True,
            )
            print(f"Experiment {i+1} completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"Experiment {i+1} failed with error: {e}")
        finally:
            # Clean up temp config
            Path(temp_config_path).unlink(missing_ok=True)
    
    print("\nAll experiments completed!")


if __name__ == "__main__":
    main()

