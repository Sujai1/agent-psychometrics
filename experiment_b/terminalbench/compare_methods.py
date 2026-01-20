#!/usr/bin/env python3
"""Compare methods for TerminalBench frontier task difficulty prediction.

Entry point for running experiment B on TerminalBench 2.0.

Usage:
    python -m experiment_b.terminalbench.compare_methods
    python -m experiment_b.terminalbench.compare_methods --verbose
"""

import sys

# Re-use the main compare_methods logic with dataset defaulted to terminalbench
if __name__ == "__main__":
    # Inject --dataset terminalbench if not specified
    if "--dataset" not in sys.argv:
        sys.argv.insert(1, "--dataset")
        sys.argv.insert(2, "terminalbench")

    from experiment_b.compare_methods import main
    main()
