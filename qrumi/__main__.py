"""Entrypoint for the CLI."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace

from qrumi import __version__


def parse_args() -> Namespace:
    """Parse command line arguments and return as dictionary."""
    parser = ArgumentParser(
        prog="modelname",
        description="Predict a target brain graph using a source brain graph.",
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    main_args = parser.add_argument_group("main options")
    infer_args = parser.add_argument_group("test options")
    data_args = parser.add_argument_group("dataset options")
    eval_args = parser.add_argument_group("evaluation options")
    main_args.add_argument(
        "-m",
        "--model-name",
        type=str,
        default="default_model_name",
        help="model name to be loaded and used for testing/inference.",
    )

    return parser.parse_args()


def main() -> None:
    """Run main function from CLI."""
    args = parse_args()



if __name__ == "__main__":
    main()
