#!/usr/bin/env python
"""Run all 4 experiments and display results tables.

Usage:
    # Run with default embeddings
    python run_all_experiments.py

    # Run with custom embeddings (applies to all experiments)
    python run_all_experiments.py --embeddings_path path/to/embeddings.npz

    # Run with different embeddings per dataset
    python run_all_experiments.py \
        --swebench_embeddings path/to/swebench_embeddings.npz \
        --terminalbench_embeddings path/to/terminalbench_embeddings.npz
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_experiment(cmd: list[str], name: str) -> str:
    """Run an experiment and return its output."""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}\n")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent,
    )

    if result.returncode != 0:
        print(f"ERROR in {name}:")
        print(result.stderr)
        return ""

    return result.stdout


def extract_summary(output: str, exp_type: str) -> str:
    """Extract the summary table from experiment output."""
    lines = output.split("\n")

    if exp_type == "a":
        # Look for "SUMMARY" section
        in_summary = False
        summary_lines = []
        for line in lines:
            if "SUMMARY" in line:
                in_summary = True
            if in_summary:
                summary_lines.append(line)
                if line.strip() == "" and len(summary_lines) > 5:
                    break
        return "\n".join(summary_lines)
    else:
        # Look for "COMPARISON TABLE" section
        in_table = False
        table_lines = []
        for line in lines:
            if "COMPARISON TABLE" in line or "EXPERIMENT B:" in line:
                in_table = True
            if in_table:
                table_lines.append(line)
        return "\n".join(table_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Run all 4 experiments and display results"
    )
    parser.add_argument(
        "--embeddings_path",
        type=str,
        help="Path to embeddings (applies to SWE-bench experiments)",
    )
    parser.add_argument(
        "--swebench_embeddings",
        type=str,
        help="Path to SWE-bench embeddings",
    )
    parser.add_argument(
        "--terminalbench_embeddings",
        type=str,
        help="Path to TerminalBench embeddings",
    )
    args = parser.parse_args()

    # Build commands
    exp_a_swebench_cmd = [sys.executable, "-m", "experiment_a.train_evaluate"]
    exp_a_terminal_cmd = [sys.executable, "-m", "experiment_a_terminalbench.train_evaluate"]
    exp_b_swebench_cmd = [sys.executable, "-m", "experiment_b.compare_methods", "--dataset", "swebench"]
    exp_b_terminal_cmd = [sys.executable, "-m", "experiment_b.compare_methods", "--dataset", "terminalbench"]

    # Add embeddings paths if specified
    swebench_emb = args.swebench_embeddings or args.embeddings_path
    terminal_emb = args.terminalbench_embeddings or args.embeddings_path

    if swebench_emb:
        exp_a_swebench_cmd.extend(["--embeddings_path", swebench_emb])
        # Note: experiment_b uses embeddings from dataset config, would need code change to override

    if terminal_emb:
        exp_a_terminal_cmd.extend(["--embeddings_path", terminal_emb])

    # Run all experiments
    results = {}

    out = run_experiment(exp_a_swebench_cmd, "Experiment A: SWE-bench")
    results["exp_a_swebench"] = extract_summary(out, "a")

    out = run_experiment(exp_a_terminal_cmd, "Experiment A: TerminalBench")
    results["exp_a_terminal"] = extract_summary(out, "a")

    out = run_experiment(exp_b_swebench_cmd, "Experiment B: SWE-bench")
    results["exp_b_swebench"] = extract_summary(out, "b")

    out = run_experiment(exp_b_terminal_cmd, "Experiment B: TerminalBench")
    results["exp_b_terminal"] = extract_summary(out, "b")

    # Print all results
    print("\n" + "="*80)
    print("ALL RESULTS")
    print("="*80)

    print("\n" + "-"*60)
    print("EXPERIMENT A: SWE-bench (5-fold CV)")
    print("-"*60)
    print(results["exp_a_swebench"])

    print("\n" + "-"*60)
    print("EXPERIMENT A: TerminalBench (5-fold CV)")
    print("-"*60)
    print(results["exp_a_terminal"])

    print("\n" + "-"*60)
    print("EXPERIMENT B: SWE-bench (Frontier Tasks)")
    print("-"*60)
    print(results["exp_b_swebench"])

    print("\n" + "-"*60)
    print("EXPERIMENT B: TerminalBench (Frontier Tasks)")
    print("-"*60)
    print(results["exp_b_terminal"])


if __name__ == "__main__":
    main()
