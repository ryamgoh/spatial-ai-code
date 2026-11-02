"""Prepare, verify, train or evaluate explicit final-v1 runs using the GPU lock."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from generation_policy import workflow_config

from spatial.v2.entailment_sft import main

if __name__ == "__main__":
    main(workflow_config())
