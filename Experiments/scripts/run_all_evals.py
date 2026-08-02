#!/usr/bin/env python3
"""Run all evaluations for every model and dataset defined in `scripts.config`.

Run this inside a tmux session to watch live output. A concise status
line is appended to `evaluate.log` after each evaluation finishes.
"""
from datetime import datetime, timezone
import subprocess
import time
import sys
import argparse

from scripts.config import DATASETS

LOG_FILE = "evaluate.log"


def run_all(datasets=None, models=None, batch_size=None, all_eval_types=False):
    target_datasets = datasets if datasets else list(DATASETS.keys())
    total = len(target_datasets)
    print(f"Running {total} dataset evaluations...")
    for dataset_key in target_datasets:
        cmd = [sys.executable, "-m", "scripts.evaluate", "--dataset", dataset_key]
        if batch_size is not None:
            cmd.extend(["--batch-size", str(batch_size)])
        if models:
            cmd.extend(["--models"] + models)
        if all_eval_types:
            cmd.append("--all-eval-types")
            
        print(f"\n=== START: dataset={dataset_key} ===")
        start = time.time()
        # Log concise START line
        ts_start = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with open(LOG_FILE, "a") as f:
            f.write(f"{ts_start} START dataset={dataset_key}\n")

        # Run and let output stream to the terminal (tmux), but capture it for error reporting
        from collections import deque
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        last_output = deque(maxlen=20)
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if line.strip():
                last_output.append(line.strip())
                
        proc.wait()

        end = time.time()
        rc = proc.returncode
        duration = end - start
        ts_end = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        
        if rc != 0:
            error_reason = " | ".join(last_output)
            entry = f"{ts_end} DONE dataset={dataset_key} exit_code={rc} duration={duration:.1f}s ERROR: {error_reason}\n"
        else:
            entry = f"{ts_end} DONE dataset={dataset_key} exit_code={rc} duration={duration:.1f}s\n"

        # Append only the concise completion line to the log
        with open(LOG_FILE, "a") as f:
            f.write(entry)
        print(f"=== DONE: {entry.strip()} ===")


def main():
    parser = argparse.ArgumentParser(description="Run evaluations across datasets and models.")
    parser.add_argument("--datasets", "-d", nargs="+", help="Datasets to evaluate (default: all)")
    parser.add_argument("--models", "-m", nargs="+", help="Models to evaluate (default: all)")
    parser.add_argument("--batch-size", "-b", type=int, default=None, help="Batch size override (default: use model specs)")
    parser.add_argument("--all-eval-types", action="store_true", help="Force rerunning all enabled eval variants")
    
    # Allow backward compatibility for old pure batch-size positional arg
    args, unknown = parser.parse_known_args()
    if unknown and len(unknown) == 1 and unknown[0].isdigit():
        args.batch_size = int(unknown[0])
        
    run_all(
        datasets=args.datasets,
        models=args.models,
        batch_size=args.batch_size,
        all_eval_types=args.all_eval_types,
    )


if __name__ == "__main__":
    main()
