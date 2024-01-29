"""Pydantic models for config parameter validation."""

from pydantic import BaseModel, conlist
from typing import List, Literal


class InputOutputConfig(BaseModel):
    input_path: str
    output_path: str
    save_output: bool
    plot_output: bool


class InitialEstimates(BaseModel):
    mumax0: float
    YXS0: float
    KS0: float
    px0: float
    pmu0: float


class Parameters(BaseModel):
    initial_estimates: InitialEstimates
    bounds: List[
        conlist(item_type=float,
                min_length=2,
                max_length=2)
    ]


class ModelConfig(BaseModel):
    estimation_mode: Literal["diff_evol", "opt_min"]
    parameters: Parameters
    

class ParameterConfig(BaseModel):
    model: ModelConfig
    input_output: InputOutputConfig
