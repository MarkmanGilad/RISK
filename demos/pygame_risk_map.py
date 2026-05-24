from __future__ import annotations

import argparse

from demos.graphics_loop import run_map_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the Risk map resource in pygame")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=815)
    parser.add_argument("--save", type=str, default="")
    parser.add_argument("--layout", choices=("full", "gameplay"), default="full")
    args = parser.parse_args()

    run_map_demo(width=args.width, height=args.height, save_path=args.save, layout=args.layout)


if __name__ == "__main__":
    main()
