import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "logs"


def run_task(task: str, gpu: str, num_workers: int):
    log_path = LOG_DIR / f"{task}.log"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "run_eval.py"),
        "--task",
        task,
        "--gpu",
        gpu,
        "--num-workers",
        str(num_workers),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu

    print(f"Running {task} on GPU {gpu}")
    print(f"Log file: {log_path}")

    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in process.stdout:
            print(line, end="")
            log_file.write(line)

        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"{task} failed with exit code {return_code}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shadow-gpu", default="0")
    parser.add_argument("--camouflage-gpu", default="0")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    LOG_DIR.mkdir(exist_ok=True)
    run_task("shadow", args.shadow_gpu, args.num_workers)
    run_task("camouflage", args.camouflage_gpu, args.num_workers)
    print("All evaluations finished successfully.")


if __name__ == "__main__":
    main()
