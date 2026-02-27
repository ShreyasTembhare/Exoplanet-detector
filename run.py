#!/usr/bin/env python3
"""
Unified launcher for the TESS exoplanet pipeline.
Auto-detects hardware (CUDA / MPS / CPU) and dispatches to the right command.

Usage:
    python run.py scan      "TIC 441462736"                   # single star
    python run.py scan      "TIC 441462736" --predict          # with classifier
    python run.py hunt      --sector 15 --limit 100            # sector sweep
    python run.py train     --data auto --epochs 30            # train model
    python run.py autopilot                                    # fully autonomous
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _hw_report():
    from device_util import print_hw_report
    print_hw_report()


def cmd_scan(argv):
    """Run the pipeline on a single TIC target."""
    _hw_report()
    sys.argv = ["run_pipeline.py"] + argv
    from run_pipeline import main
    return main()


def cmd_hunt(argv):
    """Run the autonomous sector hunter."""
    _hw_report()
    sys.argv = ["hunter.py"] + argv
    from hunter import main
    return main()


def cmd_train(argv):
    """Train or fine-tune the ResNet-1D classifier."""
    _hw_report()
    sys.argv = ["train_classifier.py"] + argv
    from train_classifier import main
    return main()


def cmd_autopilot(argv):
    """Run fully autonomous multi-sector exoplanet hunting."""
    sys.argv = ["autopilot.py"] + argv
    from autopilot import main
    return main()


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(
            "TESS Exoplanet Detector — unified launcher\n"
            "\n"
            "Usage: python run.py <command> [options]\n"
            "\n"
            "Commands:\n"
            "  scan       Analyze a single TIC target        (e.g. run.py scan \"TIC 441462736\")\n"
            "  hunt       Sweep a TESS sector autonomously   (e.g. run.py hunt --sector 15)\n"
            "  train      Train / fine-tune the classifier   (e.g. run.py train --data auto)\n"
            "  autopilot  Run fully autonomous hunting        (e.g. run.py autopilot)\n"
            "\n"
            "Run  python run.py <command> --help  for command-specific options."
        )
        sys.exit(0)

    command = sys.argv[1]
    remaining = sys.argv[2:]

    dispatch = {
        "scan": cmd_scan,
        "hunt": cmd_hunt,
        "train": cmd_train,
        "autopilot": cmd_autopilot,
    }

    if command not in dispatch:
        print(f"Unknown command: {command!r}. Choose from: scan, hunt, train, autopilot")
        sys.exit(1)

    return dispatch[command](remaining)


if __name__ == "__main__":
    main()
