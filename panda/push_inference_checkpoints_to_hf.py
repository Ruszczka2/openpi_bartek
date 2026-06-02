"""Upload inference-ready OpenPI checkpoints to Hugging Face Hub (model repo).

Uploads only params/, assets/, and _CHECKPOINT_METADATA per step (no train_state/).

Example:
  uv run python panda/push_inference_checkpoints_to_hf.py \
    --exp-name run1 \
    --repo-id Ruszczka/pi05_panda_multi_run1_reload_droid
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

from huggingface_hub import HfApi


DEFAULT_CONFIG = "pi05_panda_multi"
DEFAULT_STEPS = (200, 400, 600, 800, 999)


def _checkpoint_root(config_name: str, exp_name: str, checkpoint_base: pathlib.Path) -> pathlib.Path:
    return checkpoint_base / config_name / exp_name


def _upload_step(api: HfApi, *, local_step: pathlib.Path, repo_id: str, step: int, private: bool) -> None:
    if not local_step.is_dir():
        raise FileNotFoundError(f"Missing checkpoint step directory: {local_step}")

    for name in ("params", "assets"):
        path = local_step / name
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        print(f"  uploading {step}/{name}/ ...")
        api.upload_folder(
            folder_path=str(path),
            path_in_repo=f"{step}/{name}",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add step {step} {name}",
        )

    meta = local_step / "_CHECKPOINT_METADATA"
    if not meta.is_file():
        raise FileNotFoundError(f"Missing {meta}")
    print(f"  uploading {step}/_CHECKPOINT_METADATA ...")
    api.upload_file(
        path_or_fileobj=str(meta),
        path_in_repo=f"{step}/_CHECKPOINT_METADATA",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Add step {step} metadata",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--repo-id", required=True, help="HF model repo, e.g. Ruszczka/my_checkpoint_run1")
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=list(DEFAULT_STEPS),
        help="Checkpoint steps to upload (default: 200 400 600 800 999)",
    )
    parser.add_argument("--checkpoint-base-dir", default="checkpoints", type=pathlib.Path)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--create-repo", action="store_true", default=True)
    args = parser.parse_args()

    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        print("Warning: HF_TOKEN not set; login via `huggingface-cli login` or export HF_TOKEN.", file=sys.stderr)

    root = _checkpoint_root(args.config_name, args.exp_name, args.checkpoint_base_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint experiment dir not found: {root}")

    api = HfApi()
    if args.create_repo:
        api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    wandb_id = root / "wandb_id.txt"
    if wandb_id.is_file():
        print("uploading wandb_id.txt ...")
        api.upload_file(
            path_or_fileobj=str(wandb_id),
            path_in_repo="wandb_id.txt",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Add wandb_id.txt",
        )

    available = [s for s in args.steps if (root / str(s)).is_dir()]
    missing = [s for s in args.steps if s not in available]
    if missing:
        print(f"Skipping missing steps (not found in {root}): {missing}")
    if not available:
        raise FileNotFoundError(f"No requested checkpoint steps found in {root}. Requested: {args.steps}")

    print(f"Uploading {args.exp_name} steps {available} -> hf://{args.repo_id}")
    for step in available:
        print(f"step {step}:")
        _upload_step(api, local_step=root / str(step), repo_id=args.repo_id, step=step, private=args.private)

    print(f"Done: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
