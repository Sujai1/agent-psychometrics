#!/usr/bin/env python3
"""Compare methods for SWE-bench frontier task difficulty prediction.

Entry point for running experiment B on SWE-bench Verified.

Usage:
    python -m experiment_appendix_h_hard_tasks.swebench.compare_methods
    python -m experiment_appendix_h_hard_tasks.swebench.compare_methods --verbose
"""

import sys

# Re-use the main compare_methods logic with dataset defaulted to swebench
if __name__ == "__main__":
    # Inject --dataset swebench if not specified
    if "--dataset" not in sys.argv:
        sys.argv.insert(1, "--dataset")
        sys.argv.insert(2, "swebench")

    from experiment_appendix_h_hard_tasks.compare_methods import main
    main()
