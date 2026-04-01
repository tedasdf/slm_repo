from slm.experiments.scaling_law import ScalingLawExperiment
from slm.training.run_config import RunConfig
from slm.utils.config import load_run_config

def main():
    cfg: RunConfig = load_run_config("configs/train/scaling_base.yaml")
    experiment = ScalingLawExperiment(cfg)
    experiment.run()

if __name__ == "__main__":
    main()