"""Debug parse failures in auditor agent logs."""
from inspect_ai.log import read_eval_log
from pathlib import Path

FAILED_IDS = [
    "numpy__numpy-1fcda82", "numpy__numpy-330057f", "numpy__numpy-567b57d",
    "numpy__numpy-68eead8", "numpy__numpy-7ff7ec7", "numpy__numpy-8dd6761",
    "numpy__numpy-cb0d7cd", "numpy__numpy-cb461ba", "abetlen__llama-cpp-python-2bc1d97",
    "huggingface__transformers-253f9a3", "huggingface__transformers-63b90a5",
    "huggingface__transformers-d51b589", "pandas-dev__pandas-191557d",
    "pandas-dev__pandas-2cdca01", "pandas-dev__pandas-2f4c93e",
    "pandas-dev__pandas-45f0705", "pandas-dev__pandas-71c94af",
    "pandas-dev__pandas-9a6c8f0", "pandas-dev__pandas-ad3f3f7",
    "python-pillow__Pillow-f854676", "tornadoweb__tornado-1b464c4",
    "tornadoweb__tornado-ac13ee5", "uploadcare__pillow-simd-7511039",
    "uploadcare__pillow-simd-b4045cf",
]

log_dir = Path("chris_output/auditor_features/gso_v4_gpt54")
for log_path in sorted(log_dir.rglob("*.eval")):
    log = read_eval_log(str(log_path))
    for sample in log.samples or []:
        sid = str(sample.id)
        if any(fid in sid for fid in FAILED_IDS):
            print(f"\n=== {sid} ===")
            print(f"Stop reason: {getattr(sample.output, 'stop_reason', 'N/A')}")
            print(f"Num messages: {len(sample.messages)}")
            comp = (sample.output.completion or "")[:500] if sample.output else ""
            print(f"Completion: {comp}")
