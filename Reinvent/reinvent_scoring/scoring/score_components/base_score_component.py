from abc import ABC, abstractmethod
from typing import List

from reinvent_chemistry.conversions import Conversions

from reinvent_scoring.scoring.enums import TransformationTypeEnum, TransformationParametersEnum
from reinvent_scoring.scoring.score_transformations import TransformationFactory
from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_summary import ComponentSummary
from reinvent_scoring.scoring.enums import ComponentSpecificParametersEnum


class BaseScoreComponent(ABC):

    def __init__(self, parameters: ComponentParameters):
        self.component_specific_parameters = ComponentSpecificParametersEnum()
        self.parameters = parameters
        self._chemistry = Conversions()
        self._transformation_function = self._assign_transformation(self.parameters.specific_parameters)

    @abstractmethod
    def calculate_score(self, molecules: List, step=-1) -> ComponentSummary:
        raise NotImplementedError("calculate_score method is not implemented")

    def calculate_score_for_step(self, molecules: List, step=-1) -> ComponentSummary:
        return self.calculate_score(molecules)

    def _assign_transformation(self, specific_parameters: {}):
        transformation_type = TransformationTypeEnum()
        factory = TransformationFactory()
        if not self.parameters.specific_parameters: #FIXME: this is a hack
            self.parameters.specific_parameters = {}
        transform_params = self.parameters.specific_parameters.get(
            self.component_specific_parameters.TRANSFORMATION, {})
        if transform_params:
            transform_function = factory.get_transformation_function(transform_params)
        else:
            self.parameters.specific_parameters[
                self.component_specific_parameters.TRANSFORMATION] = {
                    TransformationParametersEnum.TRANSFORMATION_TYPE: transformation_type.NO_TRANSFORMATION
                }
            transform_function = factory.no_transformation
        return transform_function