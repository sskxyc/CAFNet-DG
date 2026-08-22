from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def _stringify(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return str(value)
    return str(value)


def _build_command(config_path: Path, seed: int | None, result_prefix: str | None, extra_args: list[str]) -> list[str]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    python = payload.get("python", "python")
    entry_script = payload["entry_script"]
    args = dict(payload.get("args", {}))

    resolved_seed = seed if seed is not None else args.get("seed")
    if resolved_seed is None:
        resolved_seed = 42

    cmd = [python, entry_script]
    for key, value in args.items():
        if key == "seed":
            value = resolved_seed
        if key == "result_prefix" and result_prefix is not None:
            value = result_prefix
        if isinstance(value, str) and "${seed}" in value:
            value = value.replace("${seed}", str(resolved_seed))

        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
            continue

        cmd.extend([f"--{key}", _stringify(value)])

    cmd.extend(extra_args)
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CAFNet-DG warm-start training/evaluation from a JSON config.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "cafnet_d_warm_main.json",
        help="Path to JSON config.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override seed in config.")
    parser.add_argument("--result-prefix", type=str, default=None, help="Override result_prefix.")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="Extra arguments passed to warm-scence.py.")
    args = parser.parse_args()

    cmd = _build_command(args.config, args.seed, args.result_prefix, args.extra)
    print(f"[train_warm] command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
