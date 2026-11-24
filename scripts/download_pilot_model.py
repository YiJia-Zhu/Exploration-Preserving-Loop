"""Download the pinned Ouro checkpoint for this project."""
import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
PROJECT_MODEL = ROOT / "models/Ouro-1.4B-Thinking"
DEFAULT_MODEL_DIR = ROOT.parent / "huggingface_models" / PROJECT_MODEL.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, help="Directory to link into models/ (defaults to ../huggingface_models/Ouro-1.4B-Thinking)")
    args = parser.parse_args()
    model_dir = (args.model_dir or DEFAULT_MODEL_DIR).resolve()
    if PROJECT_MODEL.is_symlink():
        target = PROJECT_MODEL.resolve()
        if args.model_dir is None and PROJECT_MODEL.is_dir():
            model_dir = target
        elif model_dir != target:
            parser.error(f"Relink {PROJECT_MODEL} before downloading into {model_dir}; it points to {target}")
    elif PROJECT_MODEL.exists():
        parser.error(f"Move {PROJECT_MODEL} to shared storage and link it into models/")
    if model_dir.is_relative_to(PROJECT_MODEL.parent):
        parser.error("The download directory must be outside models/")

    snapshot_download(
        "ByteDance/Ouro-1.4B-Thinking",
        revision="3aaa2224253a92ca45cf2e3d427c360e1ef9c93d",
        local_dir=model_dir,
        allow_patterns=["*.json", "*.py", "*.safetensors", "*.txt", "README.md"],
        token=False,
        max_workers=4,
    )
    PROJECT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    try:
        PROJECT_MODEL.symlink_to(os.path.relpath(model_dir, PROJECT_MODEL.parent), target_is_directory=True)
    except FileExistsError:
        if not PROJECT_MODEL.is_symlink() or PROJECT_MODEL.resolve() != model_dir:
            raise
    print(f"Project model: {PROJECT_MODEL.resolve()}")


if __name__ == "__main__":
    main()
