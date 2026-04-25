"""Run pd_ocr_trainer CLI from command line."""

from .train_recog import main, parse_args

if __name__ == "__main__":
    args = parse_args()
    main(args)
