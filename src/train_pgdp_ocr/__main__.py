"""Run train_pgdp_ocr CLI from command line."""

from .trainer import parse_args, main

if __name__ == "__main__":
    args = parse_args()
    main(args)
