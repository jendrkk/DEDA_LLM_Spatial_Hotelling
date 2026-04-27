"""Simulation orchestration: engine, batch runner, Parquet recorder."""
from hotelling.simulation.phases import Phase0BurnIn, Phase1Entry, Phase2StrategicGame
from hotelling.simulation.triggers import (
    ProfitDropTrigger,
    RivalEventTrigger,
    TimeTrigger,
    Trigger,
)

__all__ = [
    "Phase0BurnIn",
    "Phase1Entry",
    "Phase2StrategicGame",
    "ProfitDropTrigger",
    "RivalEventTrigger",
    "TimeTrigger",
    "Trigger",
]
