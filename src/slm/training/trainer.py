import torch

import wandb
from omegaconf import OmegaConf


class Trainer():
    def __init__(
            self,
            smoke_test,
            model,
            tokenizer,
            opt,
            scheduler,

                 ):
        
        self.smoke_test = smoke_test
        self.model = model
        self.opt = opt
        self.tokenizer = tokenizer

    def training_loop(self):
        raise ValueError





def main(parser):

    # loaded_cfg = OmegaConf.load(parser.config_path)
    

    # torch.manual_seed(cfg.seed)
    # random.seed(cfg.seed)
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NTP Transformer Training Pipeline")

    parser.add_argument(
        "--config_path",
        type=str,
        default="main/config/base.yaml",
        help="Config path",
    )

    parser.add_argument(
        "--smoke-test", action="store_true", help="Run a quick 1-batch validation"
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Enable W&B sweep mode and override config from wandb.config",
    )

    parser.add_argument(
        "--resume", action="store_true", help="Resume previously stored checkpoint"
    )

    parser = parser.parse_args()

    main(parser)
