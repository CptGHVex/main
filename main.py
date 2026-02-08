from __future__ import annotations

import argparse

from ui import TakeoffApp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concrete takeoff prototype")
    parser.add_argument("--pdf", type=str, help="Path to PDF")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = TakeoffApp(pdf_path=args.pdf)
    app.exec()


if __name__ == "__main__":
    main()
