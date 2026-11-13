"""Train the paper-aligned Ouro baseline or the 2026-11-13 stage scheme."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    parser.add_argument("--smoke", action="store_true", help="Random tiny official architecture, CPU semantic check only")
    parser.add_argument("--show-config", action="store_true", help="Resolve/validate configuration without loading models")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-updates", type=int)
    args = parser.parse_args()
    if args.show_config:
        from ep_loop.config import load_config, smoke_config, validate_config
        config = load_config(args.config)
        print(json.dumps(validate_config(smoke_config(config) if args.smoke else config), ensure_ascii=False, indent=2))
        return
    if not args.output:
        parser.error("--output is required for training")
    if args.smoke and args.device != "cpu":
        parser.error("Use --device cpu with --smoke")
    if args.stop_after_updates is not None and args.stop_after_updates < 1:
        parser.error("--stop-after-updates must be positive")
    from ep_loop.training import train
    train(args.config, args.output, args.device, args.smoke, args.resume, args.stop_after_updates)


if __name__ == "__main__":
    main()
