from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="open-pi-mem")
    parser.add_argument("--version", action="store_true")
    args = parser.parse_args()
    if args.version:
        print("open-pi-mem 0.1.0")
        return
    parser.print_help()


if __name__ == "__main__":
    main()
