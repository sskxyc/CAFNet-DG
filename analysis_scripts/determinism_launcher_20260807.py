from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["strict", "warn"], required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("model_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", "42")
    import torch

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=args.mode == "warn")

    script = args.script.resolve()
    model_args = list(args.model_args)
    if model_args and model_args[0] == "--":
        model_args = model_args[1:]
    manifest = {
        "mode": args.mode,
        "script": str(script),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "PYTHONHASHSEED": os.environ["PYTHONHASHSEED"],
        "arguments": model_args,
    }
    print("DETERMINISM_MANIFEST=" + json.dumps(manifest, sort_keys=True))
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script), *model_args]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
