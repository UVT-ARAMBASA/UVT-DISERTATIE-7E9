from __future__ import annotations  # ENABLE MODERN TYPE HINTS

# ================================ IMPORTS ==================================
import argparse  # CLI

from experiment_common import pick_device  # DEVICE
from run_single_matrix import run_single_matrix  # AE+DMD SINGLE
from run_multi_matrix import run_multi_matrix  # AE+DMD MULTI
from extra_optional import (  # BASELINES
    run_ae_only_single,  # AE ONLY SINGLE
    run_ae_only_multi,  # AE ONLY MULTI
    run_dmd_only_single,  # DMD ONLY SINGLE
    run_dmd_only_multi,  # DMD ONLY MULTI
)


# ================================= MAIN =====================================
def main() -> None:  # ENTRYPOINT
    parser = argparse.ArgumentParser(description="Run one dissertation experiment")  # PARSER
    parser.add_argument(  # EXPERIMENT ARG
        "experiment",
        choices=[
            "single-matrix",  # AE+DMD ONE MATRIX
            "multi-matrix",  # AE+DMD MANY MATRICES
            "ae-only-single",  # AE BASELINE ONE MATRIX
            "ae-only-multi",  # AE BASELINE MANY MATRICES
            "dmd-only-single",  # RAW DMD ONE MATRIX
            "dmd-only-multi",  # RAW DMD MANY MATRICES
        ],
        help="choose exactly one experiment to run",
    )
    args = parser.parse_args()  # PARSE

    device = pick_device()  # DEVICE
    print("DEVICE:", device)  # PRINT

    if args.experiment == "single-matrix":  # AE+DMD SINGLE
        run_single_matrix(device)  # RUN
    elif args.experiment == "multi-matrix":  # AE+DMD MULTI
        run_multi_matrix(device)  # RUN
    elif args.experiment == "ae-only-single":  # AE ONLY SINGLE
        run_ae_only_single(device)  # RUN
    elif args.experiment == "ae-only-multi":  # AE ONLY MULTI
        run_ae_only_multi(device)  # RUN
    elif args.experiment == "dmd-only-single":  # DMD ONLY SINGLE
        run_dmd_only_single(device)  # RUN
    elif args.experiment == "dmd-only-multi":  # DMD ONLY MULTI
        run_dmd_only_multi(device)  # RUN


if __name__ == "__main__":  # DIRECT RUN
    main()  # RUN
