# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "transformers==5.16.1",
#     "tokenizers==0.23.1",
#     "jinja2==3.1.6",
#     "pyyaml==6.0.3",
#     "z3-solver==4.16.0.0",
#     "scipy==1.18.1",
#     "huggingface-hub==1.29.0",
# ]
# ///
"""Final-v1 command entry point; shared implementation lives in spatial.v2."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from generation_policy import workflow_config

from spatial.v2.entailment_generation import main

if __name__ == "__main__":
    main(workflow_config())
