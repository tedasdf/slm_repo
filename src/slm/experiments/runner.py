from __future__ import annotations

from typing import Any

from slm.training import Trainer
from slm.training.config import TrainerConfig

from .base import BaseExperiment, ExperimentArtifacts


def run_experiment(
    experiment: BaseExperiment,
    trainer_cfg: TrainerConfig,
) -> Any:
    model = experiment.build_model()
    optimizer = experiment.build_optimizer(model)
    scheduler = experiment.build_scheduler(optimizer)

    artifacts: ExperimentArtifacts = experiment.build_dataloaders()
    test_cases = experiment.generate_test_cases()
    callbacks = experiment.build_callbacks()
    loss_fn = experiment.build_loss_fn()
    metrics = experiment.build_metrics()

    artifacts.test_cases.update(test_cases)
    artifacts.extra_state["metrics"] = metrics

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=artifacts.train_loader,
        val_loader=artifacts.val_loader,
        config=trainer_cfg,
        callbacks=callbacks,
        loss_fn=loss_fn,
        experiment=experiment,
    )

    trainer.state.extra["test_cases"] = artifacts.test_cases
    trainer.state.extra["metrics"] = metrics
    trainer.state.extra["experiment_artifacts"] = artifacts.extra_state

    state = trainer.train()
    return state