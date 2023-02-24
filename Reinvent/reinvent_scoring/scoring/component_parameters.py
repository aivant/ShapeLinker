from dataclasses import dataclass


@dataclass
class ComponentParameters:
    component_type: str
    name: str
    weight: float
    specific_parameters: dict = None
