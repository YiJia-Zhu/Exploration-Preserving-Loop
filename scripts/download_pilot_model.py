"""Download the pinned checkpoint to shared storage and link it into this project."""
import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
SHARED_MODEL = Path("/home/zhenning/data/huggingface_models/Ouro-1.4B-Thinking")
PROJECT_LINK = ROOT / "models/Ouro-1.4B-Thinking"


def main():
    if PROJECT_LINK.is_symlink():
        if PROJECT_LINK.resolve() != SHARED_MODEL.resolve():
            raise ValueError(f"Project link points to a different model: {PROJECT_LINK}")
    elif PROJECT_LINK.exists():
        raise FileExistsError(f"Move the existing model to shared storage first: {PROJECT_LINK}")

    SHARED_MODEL.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        "ByteDance/Ouro-1.4B-Thinking",
        revision="3aaa2224253a92ca45cf2e3d427c360e1ef9c93d",
        local_dir=SHARED_MODEL,
        allow_patterns=["*.json", "*.py", "*.safetensors", "*.txt", "README.md"],
        token=False,
        max_workers=4,
    )
    PROJECT_LINK.parent.mkdir(parents=True, exist_ok=True)
    if not PROJECT_LINK.is_symlink():
        PROJECT_LINK.symlink_to(SHARED_MODEL, target_is_directory=True)
    print(f"Shared model: {SHARED_MODEL}")
    print(f"Project link: {PROJECT_LINK} -> {SHARED_MODEL}")


if __name__ == "__main__":
    main()
