"""Train RAD-OPD by default, or select an explicit baseline configuration."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    from ep_loop.config import DEFAULT_CONFIG
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Defaults to the current main proposal")
    parser.add_argument("--output")
    parser.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    parser.add_argument("--smoke", action="store_true", help="Random tiny official architecture, CPU semantic check only")
    parser.add_argument("--show-config", action="store_true", help="Resolve/validate configuration without loading models")
    parser.add_argument("--check-assets", action="store_true", help="Check pinned local model and dataset without starting training")
    parser.add_argument("--resume", type=Path,help="Complete checkpoint directory, including optimizer and per-rank RNG")
    parser.add_argument("--stop-after-updates", type=int,help="Stop at this absolute completed update; retain the configured final target")
    args = parser.parse_args()
    if args.check_assets and args.smoke:
        parser.error("--check-assets checks pretrained assets; use --smoke separately")
    if args.show_config or args.check_assets:
        from ep_loop.config import assets_status, load_config, smoke_config, validate_config
        config = load_config(args.config)
        config = validate_config(smoke_config(config) if args.smoke else config)
        result = assets_status(config,Path(__file__).resolve().parents[1]) if args.check_assets else config
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.check_assets and not result["ready"]:
            raise SystemExit(2)
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
