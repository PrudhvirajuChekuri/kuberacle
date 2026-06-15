"""Unified CLI dispatcher: ``python -m kuberacle <command> [args...]``."""

import importlib
import sys

from kuberacle.cli import COMMANDS


def _print_usage() -> None:
    """Print available commands."""
    print("usage: python -m kuberacle <command> [args...]\n")
    print("commands:")
    for name in COMMANDS:
        print(f"  {name}")


def main(argv: list[str] | None = None) -> None:
    """Dispatch to a command module's ``main()``.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``). The first item is
            the command name; the rest are forwarded to that command.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        _print_usage()
        return

    name, rest = args[0], args[1:]
    module_path = COMMANDS.get(name)
    if module_path is None:
        print(f"Unknown command: {name}\n")
        _print_usage()
        sys.exit(2)

    # Re-shape argv so the command's own argparse sees only its arguments.
    sys.argv = [f"kuberacle {name}", *rest]
    module = importlib.import_module(module_path)
    module.main()


if __name__ == "__main__":
    main()
