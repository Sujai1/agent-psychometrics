"""
Binomial IRT model training for multi-attempt data (e.g., Terminal Bench with 5 attempts).

Instead of Bernoulli(p) for binary pass/fail, uses Binomial(n, p) to model
the number of successes out of n attempts.

Statistical model:
    p_ij = sigmoid(a_i * (theta_j - b_i))
    k_ij | theta_j, a_i, b_i ~ Binomial(n_ij, p_ij)

This uses all information from multiple attempts rather than collapsing to binary.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pyro
import pyro.distributions as dist
import torch
import torch.distributions.constraints as constraints
import numpy as np
import json
import pandas as pd
import argparse
from rich.console import Console
from rich.table import Table

console = Console()


def load_count_data(filepath):
    """
    Load raw count data from JSONL.

    Expected format:
    {"subject_id": "...", "responses": {"task1": {"successes": 4, "trials": 5}, ...}}

    Returns:
        subjects: tensor of subject indices
        items: tensor of item indices
        counts: tensor of success counts
        trials: tensor of trial counts
        subject_ids: list of subject ID strings
        item_ids: list of item ID strings
    """
    data_list = []
    with open(filepath, 'r') as f:
        for line in f:
            record = json.loads(line)
            data_list.append(record)

    # Build vocabularies
    subject_ids = [r['subject_id'] for r in data_list]
    subject_to_idx = {s: i for i, s in enumerate(subject_ids)}

    all_items = set()
    for r in data_list:
        all_items.update(r['responses'].keys())
    item_ids = sorted(all_items)
    item_to_idx = {item: i for i, item in enumerate(item_ids)}

    # Build observation tensors
    subjects_list = []
    items_list = []
    counts_list = []
    trials_list = []

    for r in data_list:
        subj_idx = subject_to_idx[r['subject_id']]
        for item_id, response in r['responses'].items():
            item_idx = item_to_idx[item_id]
            subjects_list.append(subj_idx)
            items_list.append(item_idx)
            counts_list.append(response['successes'])
            trials_list.append(response['trials'])

    return (
        torch.tensor(subjects_list, dtype=torch.long),
        torch.tensor(items_list, dtype=torch.long),
        torch.tensor(counts_list, dtype=torch.float),
        torch.tensor(trials_list, dtype=torch.float),
        subject_ids,
        item_ids
    )


def compute_item_accuracy(items, counts, trials, num_items):
    """Compute per-item accuracy as mean(counts)/mean(trials)."""
    item_counts = torch.zeros(num_items)
    item_trials = torch.zeros(num_items)

    for i in range(len(items)):
        item_counts[items[i]] += counts[i]
        item_trials[items[i]] += trials[i]

    # Avoid division by zero
    item_trials = torch.clamp(item_trials, min=1)
    return item_counts / item_trials


def init_difficulty_from_accuracy(accuracy, eps=1e-3):
    """
    Initialize difficulty from empirical accuracy.
    b_i ≈ -logit(accuracy_i)
    """
    # Clamp to avoid infinite logits
    acc_clamped = torch.clamp(accuracy, min=eps, max=1-eps)
    return -torch.log(acc_clamped / (1 - acc_clamped))


# ============== 1D Binomial 2PL Model ==============

class Binomial2PL:
    """1D 2PL IRT model with Binomial likelihood."""

    def __init__(self, num_items, num_subjects, device='cpu'):
        self.num_items = num_items
        self.num_subjects = num_subjects
        self.device = device

    def model(self, subjects, items, counts, trials):
        """Hierarchical 2PL model with Binomial likelihood."""
        # Hyperpriors
        mu_b = pyro.sample("mu_b", dist.Normal(0.0, 1e6))
        u_b = pyro.sample("u_b", dist.Gamma(1.0, 1.0))
        mu_theta = pyro.sample("mu_theta", dist.Normal(0.0, 1e6))
        u_theta = pyro.sample("u_theta", dist.Gamma(1.0, 1.0))
        mu_a = pyro.sample("mu_a", dist.Normal(0.0, 1e6))
        u_a = pyro.sample("u_a", dist.Gamma(1.0, 1.0))

        # Subject abilities
        with pyro.plate("thetas", self.num_subjects):
            ability = pyro.sample("theta", dist.Normal(mu_theta, 1.0 / u_theta))

        # Item parameters
        with pyro.plate("items", self.num_items):
            diff = pyro.sample("b", dist.Normal(mu_b, 1.0 / u_b))
            slope = pyro.sample("a", dist.Normal(mu_a, 1.0 / u_a))

        # Observations: Binomial instead of Bernoulli
        with pyro.plate("obs", counts.size(0)):
            logits = slope[items] * (ability[subjects] - diff[items])
            pyro.sample("y", dist.Binomial(total_count=trials, logits=logits), obs=counts)

    def guide(self, subjects, items, counts, trials):
        """Variational guide."""
        # Hyperparameter guides
        loc_mu_b = pyro.param("loc_mu_b", torch.tensor(0.0))
        scale_mu_b = pyro.param("scale_mu_b", torch.tensor(10.0), constraint=constraints.positive)
        loc_mu_theta = pyro.param("loc_mu_theta", torch.tensor(0.0))
        scale_mu_theta = pyro.param("scale_mu_theta", torch.tensor(10.0), constraint=constraints.positive)
        loc_mu_a = pyro.param("loc_mu_a", torch.tensor(0.0))
        scale_mu_a = pyro.param("scale_mu_a", torch.tensor(10.0), constraint=constraints.positive)

        alpha_b = pyro.param("alpha_b", torch.tensor(1.0), constraint=constraints.positive)
        beta_b = pyro.param("beta_b", torch.tensor(1.0), constraint=constraints.positive)
        alpha_theta = pyro.param("alpha_theta", torch.tensor(1.0), constraint=constraints.positive)
        beta_theta = pyro.param("beta_theta", torch.tensor(1.0), constraint=constraints.positive)
        alpha_a = pyro.param("alpha_a", torch.tensor(1.0), constraint=constraints.positive)
        beta_a = pyro.param("beta_a", torch.tensor(1.0), constraint=constraints.positive)

        # Per-parameter guides
        loc_ability = pyro.param("loc_ability", torch.zeros(self.num_subjects))
        scale_ability = pyro.param("scale_ability", torch.ones(self.num_subjects), constraint=constraints.positive)
        loc_diff = pyro.param("loc_diff", torch.zeros(self.num_items))
        scale_diff = pyro.param("scale_diff", torch.ones(self.num_items), constraint=constraints.positive)
        loc_slope = pyro.param("loc_slope", torch.zeros(self.num_items))
        scale_slope = pyro.param("scale_slope", torch.ones(self.num_items), constraint=constraints.positive)

        # Sample hyperparameters
        pyro.sample("mu_b", dist.Normal(loc_mu_b, scale_mu_b))
        pyro.sample("u_b", dist.Gamma(alpha_b, beta_b))
        pyro.sample("mu_theta", dist.Normal(loc_mu_theta, scale_mu_theta))
        pyro.sample("u_theta", dist.Gamma(alpha_theta, beta_theta))
        pyro.sample("mu_a", dist.Normal(loc_mu_a, scale_mu_a))
        pyro.sample("u_a", dist.Gamma(alpha_a, beta_a))

        # Sample latent variables
        with pyro.plate("thetas", self.num_subjects):
            pyro.sample("theta", dist.Normal(loc_ability, scale_ability))

        with pyro.plate("items", self.num_items):
            pyro.sample("b", dist.Normal(loc_diff, scale_diff))
            pyro.sample("a", dist.Normal(loc_slope, scale_slope))


# ============== Multidim Binomial 2PL Model ==============

class BinomialMIRT:
    """Multidimensional 2PL IRT model with Binomial likelihood."""

    def __init__(self, num_items, num_subjects, dims=2, device='cpu'):
        self.num_items = num_items
        self.num_subjects = num_subjects
        self.dims = dims
        self.device = device

    def model(self, subjects, items, counts, trials):
        """Hierarchical MIRT model with Binomial likelihood."""
        # Hyperpriors for each dimension
        with pyro.plate("mu_b_plate", self.dims):
            mu_b = pyro.sample("mu_b", dist.Normal(0.0, 1e6))
        with pyro.plate("u_b_plate", self.dims):
            u_b = pyro.sample("u_b", dist.Gamma(1.0, 1.0))
        with pyro.plate("mu_theta_plate", self.dims):
            mu_theta = pyro.sample("mu_theta", dist.Normal(0.0, 1e6))
        with pyro.plate("u_theta_plate", self.dims):
            u_theta = pyro.sample("u_theta", dist.Gamma(1.0, 1.0))
        with pyro.plate("mu_gamma_plate", self.dims):
            mu_gamma = pyro.sample("mu_gamma", dist.Normal(0.0, 1e6))
        with pyro.plate("u_gamma_plate", self.dims):
            u_gamma = pyro.sample("u_gamma", dist.Gamma(1.0, 1.0))

        # Subject abilities [num_subjects, dims]
        with pyro.plate("thetas", self.num_subjects, dim=-2):
            with pyro.plate("theta_dims", self.dims, dim=-1):
                ability = pyro.sample("theta", dist.Normal(mu_theta, 1.0 / u_theta))

        # Item parameters [num_items, dims]
        with pyro.plate("bs", self.num_items, dim=-2):
            with pyro.plate("bs_dims", self.dims, dim=-1):
                diff = pyro.sample("b", dist.Normal(mu_b, 1.0 / u_b))

        with pyro.plate("gammas", self.num_items, dim=-2):
            with pyro.plate("gamma_dims", self.dims, dim=-1):
                disc = pyro.sample("gamma", dist.Normal(mu_gamma, 1.0 / u_gamma))

        # Observations: Binomial
        with pyro.plate("obs", counts.size(0)):
            # Sum over dimensions: logits = sum_d(disc[i,d] * (ability[j,d] - diff[i,d]))
            multidim_logits = disc[items] * (ability[subjects] - diff[items])
            logits = multidim_logits.sum(dim=-1)
            pyro.sample("y", dist.Binomial(total_count=trials, logits=logits), obs=counts)

    def guide(self, subjects, items, counts, trials):
        """Variational guide."""
        # Hyperparameter guides
        loc_mu_b = pyro.param("loc_mu_b", torch.zeros(self.dims))
        scale_mu_b = pyro.param("scale_mu_b", 100 * torch.ones(self.dims), constraint=constraints.positive)
        loc_mu_theta = pyro.param("loc_mu_theta", torch.zeros(self.dims))
        scale_mu_theta = pyro.param("scale_mu_theta", 100 * torch.ones(self.dims), constraint=constraints.positive)
        loc_mu_gamma = pyro.param("loc_mu_gamma", torch.zeros(self.dims))
        scale_mu_gamma = pyro.param("scale_mu_gamma", 100 * torch.ones(self.dims), constraint=constraints.positive)

        alpha_b = pyro.param("alpha_b", torch.ones(self.dims), constraint=constraints.positive)
        beta_b = pyro.param("beta_b", torch.ones(self.dims), constraint=constraints.positive)
        alpha_theta = pyro.param("alpha_theta", torch.ones(self.dims), constraint=constraints.positive)
        beta_theta = pyro.param("beta_theta", torch.ones(self.dims), constraint=constraints.positive)
        alpha_gamma = pyro.param("alpha_gamma", torch.ones(self.dims), constraint=constraints.positive)
        beta_gamma = pyro.param("beta_gamma", torch.ones(self.dims), constraint=constraints.positive)

        # Per-parameter guides
        loc_ability = pyro.param("loc_ability", torch.zeros(self.num_subjects, self.dims))
        scale_ability = pyro.param("scale_ability", torch.ones(self.num_subjects, self.dims), constraint=constraints.positive)
        loc_diff = pyro.param("loc_diff", torch.zeros(self.num_items, self.dims))
        scale_diff = pyro.param("scale_diff", torch.ones(self.num_items, self.dims), constraint=constraints.positive)
        loc_disc = pyro.param("loc_disc", torch.zeros(self.num_items, self.dims))
        scale_disc = pyro.param("scale_disc", torch.ones(self.num_items, self.dims), constraint=constraints.positive)

        # Sample hyperparameters
        with pyro.plate("mu_b_plate", self.dims):
            pyro.sample("mu_b", dist.Normal(loc_mu_b, scale_mu_b))
        with pyro.plate("u_b_plate", self.dims):
            pyro.sample("u_b", dist.Gamma(alpha_b, beta_b))
        with pyro.plate("mu_theta_plate", self.dims):
            pyro.sample("mu_theta", dist.Normal(loc_mu_theta, scale_mu_theta))
        with pyro.plate("u_theta_plate", self.dims):
            pyro.sample("u_theta", dist.Gamma(alpha_theta, beta_theta))
        with pyro.plate("mu_gamma_plate", self.dims):
            pyro.sample("mu_gamma", dist.Normal(loc_mu_gamma, scale_mu_gamma))
        with pyro.plate("u_gamma_plate", self.dims):
            pyro.sample("u_gamma", dist.Gamma(alpha_gamma, beta_gamma))

        # Sample latent variables
        with pyro.plate("thetas", self.num_subjects, dim=-2):
            with pyro.plate("theta_dims", self.dims, dim=-1):
                pyro.sample("theta", dist.Normal(loc_ability, scale_ability))

        with pyro.plate("bs", self.num_items, dim=-2):
            with pyro.plate("bs_dims", self.dims, dim=-1):
                pyro.sample("b", dist.Normal(loc_diff, scale_diff))

        with pyro.plate("gammas", self.num_items, dim=-2):
            with pyro.plate("gamma_dims", self.dims, dim=-1):
                pyro.sample("gamma", dist.Normal(loc_disc, scale_disc))


# ============== Training Functions ==============

def train_model(model, guide, subjects, items, counts, trials, epochs=5000, lr=0.1, lr_decay=0.9999):
    """Train a Pyro model using SVI."""
    from pyro.infer import SVI, Trace_ELBO
    from pyro.optim import ClippedAdam

    optimizer = ClippedAdam({"lr": lr, "betas": (0.9, 0.999), "clip_norm": 5.0, "lrd": lr_decay})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    best_loss = float('inf')

    table = Table(title=f"Training Binomial IRT Model for {epochs} epochs")
    table.add_column("Epoch", justify="right")
    table.add_column("Loss", justify="right")
    table.add_column("Best Loss", justify="right")
    table.add_column("LR", justify="right")

    for epoch in range(1, epochs + 1):
        loss = svi.step(subjects, items, counts, trials)

        if loss < best_loss:
            best_loss = loss

        if epoch == 1 or epoch % 100 == 0 or epoch == epochs:
            current_lr = lr * (lr_decay ** epoch)
            table.add_row(
                str(epoch),
                f"{loss:.4f}",
                f"{best_loss:.4f}",
                f"{current_lr:.4f}"
            )

    console.print(table)
    return best_loss


def fit_1d_binomial(subjects, items, counts, trials, subject_ids, item_ids,
                    epochs=5000, output_dir=None):
    """Fit 1D Binomial 2PL model."""
    num_subjects = len(subject_ids)
    num_items = len(item_ids)

    # Initialize difficulty from accuracy
    accuracy = compute_item_accuracy(items, counts, trials, num_items)
    init_diff = init_difficulty_from_accuracy(accuracy)

    console.print(f"[bold]Fitting 1D Binomial 2PL[/bold]")
    console.print(f"  Subjects: {num_subjects}, Items: {num_items}")
    console.print(f"  Observations: {len(counts)}")

    # Initialize model
    model_obj = Binomial2PL(num_items, num_subjects)

    # Set initial difficulty values
    pyro.clear_param_store()
    pyro.param("loc_diff", init_diff)
    pyro.param("loc_slope", torch.ones(num_items))  # Start with a=1

    # Train
    best_loss = train_model(
        model_obj.model, model_obj.guide,
        subjects, items, counts, trials,
        epochs=epochs
    )

    # Extract results
    abilities = pyro.param("loc_ability").detach().numpy()
    difficulties = pyro.param("loc_diff").detach().numpy()
    discriminations = pyro.param("loc_slope").detach().numpy()

    ability_std = pyro.param("scale_ability").detach().numpy()
    diff_std = pyro.param("scale_diff").detach().numpy()
    disc_std = pyro.param("scale_slope").detach().numpy()

    # Save results
    if output_dir:
        out_path = Path(output_dir) / "1d"
        out_path.mkdir(parents=True, exist_ok=True)

        items_df = pd.DataFrame({
            "a": discriminations,
            "b": difficulties,
            "a_std": disc_std,
            "b_std": diff_std,
        }, index=item_ids)
        items_df.to_csv(out_path / "items.csv")

        abilities_df = pd.DataFrame({
            "theta": abilities,
            "theta_std": ability_std,
        }, index=subject_ids)
        abilities_df.sort_values("theta", ascending=False).to_csv(out_path / "abilities.csv")

        console.print(f"[green]Saved results to {out_path}[/green]")

    return best_loss, num_items + num_subjects + num_items  # approx n_params


def fit_mirt_binomial(subjects, items, counts, trials, subject_ids, item_ids,
                      dims=2, epochs=5000, output_dir=None):
    """Fit multidimensional Binomial 2PL model."""
    num_subjects = len(subject_ids)
    num_items = len(item_ids)

    # Initialize difficulty from accuracy
    accuracy = compute_item_accuracy(items, counts, trials, num_items)
    init_diff = init_difficulty_from_accuracy(accuracy)

    console.print(f"[bold]Fitting {dims}D Binomial MIRT[/bold]")
    console.print(f"  Subjects: {num_subjects}, Items: {num_items}, Dims: {dims}")
    console.print(f"  Observations: {len(counts)}")

    # Initialize model
    model_obj = BinomialMIRT(num_items, num_subjects, dims=dims)

    # Set initial values
    pyro.clear_param_store()
    # Replicate 1D init across dimensions with some jitter
    init_diff_md = init_diff.unsqueeze(1).expand(-1, dims) + 0.1 * torch.randn(num_items, dims)
    pyro.param("loc_diff", init_diff_md)
    pyro.param("loc_disc", 0.5 * torch.ones(num_items, dims))  # Start with small positive disc

    # Train with lower LR for MIRT
    best_loss = train_model(
        model_obj.model, model_obj.guide,
        subjects, items, counts, trials,
        epochs=epochs, lr=0.003, lr_decay=1.0
    )

    # Extract results
    abilities = pyro.param("loc_ability").detach().numpy()
    difficulties = pyro.param("loc_diff").detach().numpy()
    discriminations = pyro.param("loc_disc").detach().numpy()

    ability_std = pyro.param("scale_ability").detach().numpy()
    diff_std = pyro.param("scale_diff").detach().numpy()
    disc_std = pyro.param("scale_disc").detach().numpy()

    # Save results
    if output_dir:
        out_path = Path(output_dir) / f"{dims}d"
        out_path.mkdir(parents=True, exist_ok=True)

        # Items DataFrame
        item_rows = {}
        for i, item_id in enumerate(item_ids):
            row = {}
            for d in range(dims):
                row[f"a{d+1}"] = discriminations[i, d]
                row[f"b{d+1}"] = difficulties[i, d]
                row[f"a{d+1}_std"] = disc_std[i, d]
                row[f"b{d+1}_std"] = diff_std[i, d]
            item_rows[item_id] = row
        pd.DataFrame.from_dict(item_rows, orient="index").to_csv(out_path / "items.csv")

        # Abilities DataFrame
        abil_rows = {}
        for i, subj_id in enumerate(subject_ids):
            row = {}
            for d in range(dims):
                row[f"theta{d+1}"] = abilities[i, d]
                row[f"theta{d+1}_std"] = ability_std[i, d]
            abil_rows[subj_id] = row
        abilities_df = pd.DataFrame.from_dict(abil_rows, orient="index")
        abilities_df["theta_avg"] = abilities_df[[f"theta{d+1}" for d in range(dims)]].mean(axis=1)
        abilities_df.sort_values("theta_avg", ascending=False).to_csv(out_path / "abilities.csv")

        console.print(f"[green]Saved results to {out_path}[/green]")

    # Approximate n_params
    n_params = dims * (num_items * 2 + num_subjects)
    return best_loss, n_params


def main():
    parser = argparse.ArgumentParser(description='Train Binomial IRT models on multi-attempt data')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to raw JSONL with count data')
    parser.add_argument('--dims', type=int, nargs='+', default=[1, 2, 3],
                       help='Dimensions to fit (default: 1 2 3)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save results')
    parser.add_argument('--epochs', type=int, default=5000,
                       help='Number of training epochs')
    args = parser.parse_args()

    # Load data
    console.print(f"[bold]Loading data from {args.data_path}[/bold]")
    subjects, items, counts, trials, subject_ids, item_ids = load_count_data(args.data_path)

    console.print(f"  Loaded {len(subject_ids)} subjects, {len(item_ids)} items")
    console.print(f"  Total observations: {len(counts)}")
    console.print(f"  Mean trials per obs: {trials.mean():.1f}")
    console.print(f"  Overall success rate: {counts.sum() / trials.sum():.1%}")

    results = []

    for dim in args.dims:
        pyro.clear_param_store()

        if dim == 1:
            loss, n_params = fit_1d_binomial(
                subjects, items, counts, trials, subject_ids, item_ids,
                epochs=args.epochs, output_dir=args.output_dir
            )
        else:
            loss, n_params = fit_mirt_binomial(
                subjects, items, counts, trials, subject_ids, item_ids,
                dims=dim, epochs=args.epochs, output_dir=args.output_dir
            )

        n_obs = len(counts)
        aic = 2 * loss + 2 * n_params
        bic = 2 * loss + n_params * np.log(n_obs)

        results.append({
            'model': f'{dim}D',
            'loss': loss,
            'n_params': n_params,
            'n_obs': n_obs,
            'AIC': aic,
            'BIC': bic
        })

    # Print comparison
    console.print("\n[bold]Model Comparison[/bold]")
    results_df = pd.DataFrame(results)
    console.print(results_df.to_string())

    # Save comparison
    out_path = Path(args.output_dir)
    results_df.to_csv(out_path / "model_selection.csv", index=False)
    console.print(f"\n[green]Saved model comparison to {out_path / 'model_selection.csv'}[/green]")


if __name__ == "__main__":
    main()
