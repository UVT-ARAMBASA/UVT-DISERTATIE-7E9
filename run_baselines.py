from __future__ import annotations  # ENABLE MODERN TYPE HINTS

# ================================ IMPORTS ==================================
import argparse  # CLI

from experiment_common import pick_device  # DEVICE
from extra_optional import run_ae_only_baseline, run_dmd_only_baseline  # BASELINES


# ================================= MAIN =====================================
def main() -> None:  # ENTRYPOINT
    parser = argparse.ArgumentParser(description="Run optional baselines only")  # PARSER
    parser.add_argument("baseline", choices=["ae-only", "dmd-only", "both"], help="which baseline to run")  # ARG
    args = parser.parse_args()  # PARSE

    device = pick_device()  # DEVICE
    print("DEVICE:", device)  # PRINT

    if args.baseline in ("ae-only", "both"):  # AE ONLY
        run_ae_only_baseline(device)  # RUN

    if args.baseline in ("dmd-only", "both"):  # DMD ONLY
        run_dmd_only_baseline(device)  # RUN


if __name__ == "__main__":  # DIRECT RUN
    main()  # RUN