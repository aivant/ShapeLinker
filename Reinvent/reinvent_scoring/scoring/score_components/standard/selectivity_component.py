import pickle

from typing import List

from reinvent_scoring.scoring.predictive_model.base_model_container import BaseModelContainer
from reinvent_scoring.scoring.predictive_model.model_container import ModelContainer
from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_components import BaseScoreComponent
from reinvent_scoring.scoring.score_summary import ComponentSummary
from reinvent_scoring.scoring.score_transformations import TransformationFactory
from reinvent_scoring.scoring.enums import TransformationTypeEnum, TransformationParametersEnum


class SelectivityComponent(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        self._transformation_type = TransformationTypeEnum()
        self._model_transformation = self._assign_delta_transformation(
            {TransformationParametersEnum.TRANSFORMATION_TYPE: self._transformation_type.NO_TRANSFORMATION})
        self._activity_params = self._prepare_activity_parameters(parameters)
        self._off_target_params = self._prepare_offtarget_parameters(parameters)
        self._activity_model = self._load_model(self._activity_params)
        self._off_target_activity_model = self._load_model(self._off_target_params)
        self._delta_params = self._prepare_delta_parameters(parameters)
        self._delta_transformation = self._assign_delta_transformation(self._delta_params)

    def calculate_score(self, molecules: List) -> ComponentSummary:
        score, offtarget_score = self._calculate_offtarget_activity(molecules, self._activity_params,
                                                                    self._off_target_params, self._delta_params)
        score_summary = ComponentSummary(total_score=score, parameters=self._off_target_params,
                                         raw_score=offtarget_score)
        return score_summary

    def _load_model(self, parameters: ComponentParameters):
        try:
            activity_model = self._load_scikit_model(parameters)
        except Exception as e:
            model_path = self.parameters.specific_parameters.get(self.component_specific_parameters.MODEL_PATH, "")
            raise Exception(f"The loaded file `{model_path}` isn't a valid scikit-learn model: {e}.")
        return activity_model

    def _load_scikit_model(self, parameters: ComponentParameters) -> BaseModelContainer:
        model_path = parameters.specific_parameters.get(self.component_specific_parameters.MODEL_PATH, "")
        with open(model_path, "rb") as f:
            scikit_model = pickle.load(f)

            models_are_identical = self._activity_params.specific_parameters[
                                       self.component_specific_parameters.SCIKIT] == \
                                   self._off_target_params.specific_parameters[
                                       self.component_specific_parameters.SCIKIT]

            model_is_regression = self._off_target_params.specific_parameters[
                                      self.component_specific_parameters.SCIKIT] == "regression"

            both_models_are_regression = models_are_identical and model_is_regression

            if both_models_are_regression:
                parameters.specific_parameters[self.component_specific_parameters.TRANSFORMATION] = {}

            self._assign_model_transformation(both_models_are_regression)

            packaged_model = ModelContainer(scikit_model, parameters.specific_parameters)
        return packaged_model

    def _calculate_offtarget_activity(self, molecules, activity_params, offtarget_params, delta_params):
        raw_activity_score = self._activity_model\
            .predict(molecules, activity_params.specific_parameters)
        raw_offtarget_score = self._off_target_activity_model \
            .predict(molecules, offtarget_params.specific_parameters)
        activity_score = self._apply_model_transformation(raw_activity_score, activity_params.specific_parameters)
        offtarget_score = self._apply_model_transformation(raw_offtarget_score, offtarget_params.specific_parameters)

        delta = activity_score - offtarget_score

        transformed_score = self._delta_transformation(delta, delta_params) if delta_params[
            self.component_specific_parameters.TRANSFORMATION] else delta
        transformed_score[transformed_score < 0.01] = 0.01

        return transformed_score, raw_offtarget_score

    def _assign_delta_transformation(self, specific_parameters: {}):
        factory = TransformationFactory()
        transform_function = factory.get_transformation_function(specific_parameters)
        return transform_function

    def _prepare_activity_parameters(self, parameters: ComponentParameters) -> ComponentParameters:
        specific_parameters = parameters.specific_parameters["activity_specific_parameters"]
        activity_params = ComponentParameters(name=self.parameters.name,
                                              weight=self.parameters.weight,
                                              component_type=self.parameters.component_type,
                                              specific_parameters=specific_parameters
                                              )
        return activity_params

    def _prepare_offtarget_parameters(self, parameters: ComponentParameters) -> ComponentParameters:
        specific_parameters = parameters.specific_parameters["offtarget_specific_parameters"]
        offtarget_params = ComponentParameters(name=self.parameters.name,
                                               weight=self.parameters.weight,
                                               component_type=self.parameters.component_type,
                                               specific_parameters=specific_parameters
                                               )
        return offtarget_params

    def _prepare_delta_parameters(self, parameters: ComponentParameters) -> dict:
        specific_params = parameters.specific_parameters["delta_transformation_parameters"]
        specific_params[self.component_specific_parameters.TRANSFORMATION] = \
            "regression" == self._activity_params.specific_parameters[self.component_specific_parameters.SCIKIT] == \
            self._off_target_params.specific_parameters[self.component_specific_parameters.SCIKIT]
        return specific_params

    def _apply_model_transformation(self, predicted_activity, parameters: dict):
        transform_params = parameters.get(self.component_specific_parameters.TRANSFORMATION, {})
        if transform_params:
            activity = self._model_transformation(predicted_activity, transform_params)
        else:
            activity = predicted_activity
        return activity

    def _assign_model_transformation(self, both_models_are_regression: bool):
        if both_models_are_regression:
            return
        if self._activity_params.specific_parameters.get(self.component_specific_parameters.SCIKIT) == "regression":
            self._model_transformation = self._assign_delta_transformation(
                self._activity_params.specific_parameters.get(self.component_specific_parameters.TRANSFORMATION, {}))
        if self._off_target_params.specific_parameters.get(self.component_specific_parameters.SCIKIT) == "regression":
            self._model_transformation = self._assign_delta_transformation(
                self._off_target_params.specific_parameters.get(self.component_specific_parameters.TRANSFORMATION, {}))
