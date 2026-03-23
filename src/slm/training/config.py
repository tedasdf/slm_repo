from dataclasses import dataclass

@dataclass
class TrainingConfig:
    seeds: int = 1337
    