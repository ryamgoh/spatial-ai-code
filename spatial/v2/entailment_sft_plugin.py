"""Axolotl extension for final-v1's checked trainer boundary and epoch Dev callback."""

from pathlib import Path

from axolotl.integrations.base import BasePlugin

from spatial.v2.entailment_sft import verify_run
from spatial.v2.entailment_sft_runtime import EpochDevCallback


class FinalSFTPlugin(BasePlugin):
    def get_training_args(self, cfg):
        return {
            "prediction_loss_only": True,
            "data_seed": cfg.seed,
            "logging_nan_inf_filter": False,
        }

    def post_trainer_create(self, cfg, trainer):
        directory = Path(cfg.output_dir).parent
        meta = verify_run(directory)
        trainer.add_callback(EpochDevCallback(trainer, directory, meta))
