from typing import Dict, Any

from reinvent_scoring.scoring.enums.container_type_enum import ContainerType
from reinvent_scoring.scoring.enums.component_specific_parameters_enum import ComponentSpecificParametersEnum

from reinvent_scoring.scoring.predictive_model.base_model_container import BaseModelContainer
from reinvent_scoring.scoring.predictive_model.optuna_container import OptunaModelContainer
from reinvent_scoring.scoring.predictive_model.scikit_model_container import ScikitModelContainer


class ModelContainer:

    def __new__(cls, activity_model: Any, specific_parameters: Dict) -> BaseModelContainer:
        _component_specific_parameters = ComponentSpecificParametersEnum()
        _container_type = ContainerType()
        container_type = specific_parameters.get(_component_specific_parameters.CONTAINER_TYPE,
                                                 _container_type.SCIKIT_CONTAINER)
        if container_type == _container_type.SCIKIT_CONTAINER:
            container_instance = ScikitModelContainer(activity_model,
                                                      specific_parameters[_component_specific_parameters.SCIKIT],
                                                      specific_parameters)
        else:
            # TODO: possibly a good spot for error try/catching
            container_instance = OptunaModelContainer(activity_model)

        return container_instance
