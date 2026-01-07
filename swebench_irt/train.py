"""
This script trains the IRT model and allows specification the number of dimensions.
For the main analysis, we look at dimensions 1 - 6.
Call the function using the following syntax:
    python swebench_irt/train.py --dims 1 2 3 4 5 6 --output_dir training_results --epochs 5000
    python swebench_irt/train.py --data_path chris_output/clean_data/swebench_verified/swebench_verified.jsonl --dims 1 2 3
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pyro
import numpy as np
import json
import pandas as pd
from py_irt.dataset import Dataset
from py_irt.models import Multidim2PL
from py_irt.models import TwoParamLog
from py_irt.config import IrtConfig
from py_irt.training import IrtModelTrainer
import argparse

def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (ROOT / path)

def load_irt_data(filepath):
    """Load and reshape JSONL data for IRT analysis."""
    data_list = []
    with open(filepath, 'r') as f:
        for line in f:
            record = json.loads(line)
            row = {'subject_id': record['subject_id']}
            row.update(record['responses'])
            data_list.append(row)
    
    df = pd.DataFrame(data_list)
    item_columns = [col for col in df.columns if col != 'subject_id']
    return Dataset.from_pandas(df, subject_column="subject_id", item_columns=item_columns), item_columns

def fit_1d_irt(data: Dataset, epochs: int, output_name: str) -> IrtModelTrainer:
    config = IrtConfig(
        model_type=TwoParamLog,
        priors="hierarchical",
        initializers=[
            {"name": "difficulty_from_accuracy", "eps": 1e-3, "log_limit": 0},
        ],
    )
    trainer = IrtModelTrainer(config=config, data_path=None, dataset=data)
    trainer.train(epochs=epochs)

    # Extract parameters and uncertainties
    discriminations = list(trainer.best_params["disc"])
    difficulties = list(trainer.best_params["diff"])
    # Preserve alignment with parameter arrays by iterating indices in order
    item_id_map = trainer.best_params["item_ids"]  # {index:int -> item_id:str}
    subject_id_map = trainer.best_params["subject_ids"]  # {index:int -> subject_id:str}
    item_ids = [item_id_map[i] for i in range(len(difficulties))]
    abilities = list(trainer.best_params["ability"])
    subject_ids = [subject_id_map[i] for i in range(len(abilities))]

    ability_std = list(pyro.param("scale_ability").detach().cpu().numpy())
    diff_std = list(pyro.param("scale_diff").detach().cpu().numpy())
    disc_log_std = list(pyro.param("scale_slope").detach().cpu().numpy())

    out_dir = ROOT / "clean_data" / output_name / "1d"
    out_dir.mkdir(parents=True, exist_ok=True)

    items_df = pd.DataFrame({
        "a": discriminations,
        "b": difficulties,
        "a_std": disc_log_std,
        "b_std": diff_std,
    }, index=item_ids)
    items_df.to_csv(out_dir / "items.csv")

    abilities_df = pd.DataFrame({
        "theta": abilities,
        "theta_std": ability_std,
    }, index=subject_ids).sort_values("theta", ascending=False)
    abilities_df.to_csv(out_dir / "abilities.csv")

    return trainer

def fit_md_irt(data: Dataset, dims:int, epochs:int, output_name:str) -> IrtModelTrainer:

    config = IrtConfig(
        model_type=Multidim2PL,
        priors="hierarchical",
        lr=0.003,
        lr_decay=1.0,
        clip_norm=5.0,
        dims=dims,
        initializers=[
            {
                "name": "difficulty_from_accuracy",
                "eps": 1e-3,
                "dims": dims,
                "jitter_std": 0.1,
                "init_disc_std": 0.0,      # PCA will set disc/ability
                "init_ability_std": 0.0,
                "log_limit": 0,
            },
            {
                "name": "mirt_pca",
                "dims": dims,
                "disc_scale": 0.5,
                "ability_scale": 0.5,
                "center": "item",
            },
        ],
    )
    trainer = IrtModelTrainer(config=config, data_path=None, dataset=data)
    trainer.train(epochs=epochs)

    # Convert from pytorch tensor to numpy array
    abilities = pyro.param("loc_ability").detach().cpu().numpy()  # [S, D]
    difficulties = pyro.param("loc_diff").detach().cpu().numpy()  # [I, D]
        # For MIRT: discrimination uses Normal guide; do NOT exponentiate
    discriminations = pyro.param("loc_disc").detach().cpu().numpy()  # [I, D]

    ability_std = pyro.param("scale_ability").detach().cpu().numpy()
    diff_std = pyro.param("scale_diff").detach().cpu().numpy()
    disc_std = pyro.param("scale_disc").detach().cpu().numpy()

    out_dir = ROOT / "clean_data" / output_name / f"{dims}d"
    out_dir.mkdir(parents=True, exist_ok=True)

    item_id_map = trainer.best_params["item_ids"]
    subject_id_map = trainer.best_params["subject_ids"]
    item_ids = [item_id_map[i] for i in range(difficulties.shape[0])]
    subject_ids = [subject_id_map[i] for i in range(abilities.shape[0])]

    item_rows = {}
    for i_idx, iid in enumerate(item_ids):
        row = {}
        for d in range(dims):
            row[f"a{d+1}"] = discriminations[i_idx, d]
            row[f"b{d+1}"] = difficulties[i_idx, d]
            row[f"a{d+1}_std"] = disc_std[i_idx, d]
            row[f"b{d+1}_std"] = diff_std[i_idx, d]
        item_rows[iid] = row
    pd.DataFrame.from_dict(item_rows, orient="index").to_csv(out_dir / "items.csv")

    # Abilities
    abil_rows = {}
    for s_idx, sid in enumerate(subject_ids):
        row = {}
        for d in range(dims):
            row[f"theta{d+1}"] = abilities[s_idx, d]
            row[f"theta{d+1}_std"] = ability_std[s_idx, d]
        abil_rows[sid] = row
    abilities_df = pd.DataFrame.from_dict(abil_rows, orient="index")
    abilities_df["theta_avg"] = abilities_df[[f"theta{d+1}" for d in range(dims)]].mean(axis=1)
    abilities_df.sort_values("theta_avg", ascending=False).to_csv(out_dir / "abilities.csv")

    return trainer

def main():
    parser = argparse.ArgumentParser(description='Train IRT models')
    parser.add_argument('--dims', type=int, nargs='*', default=[1, 2, 3, 4, 5, 6], 
        help='Dims to fit (default: 1–6)')
    parser.add_argument('--output_dir', type=str, default="chris_output/clean_data/training_results",
        help="Directory to save results to")
    parser.add_argument('--data_path', type=str, default="chris_output/clean_data/swebench_verified/swebench_verified.jsonl",
        help="Path to JSONL responses")
    parser.add_argument('--epochs', type=int, default=5000,
        help='Number of training epochs (default: 5000)')
    args = parser.parse_args()
    
    data_path = resolve_path(args.data_path)
    data, item_columns = load_irt_data(data_path)

    for dim in args.dims:
        pyro.clear_param_store()  # Clear parameters between different dimensional models
        if dim == 1:
            fit_1d_irt(data=data, epochs=args.epochs, output_name=args.output_dir)
        else:
            fit_md_irt(data=data, dims=dim, epochs=args.epochs, output_name=args.output_dir)

if __name__ == "__main__":
    main()
