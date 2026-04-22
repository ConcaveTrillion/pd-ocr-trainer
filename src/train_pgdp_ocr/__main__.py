"""Run train_pgdp_ocr CLI from command line."""

from .trainer import main, parse_args

if __name__ == "__main__":
    args = parse_args()
    main(args)
