from reinvent_models.model_factory.configurations.model_configuration import ModelConfiguration
from reinvent_models.model_factory.enums.model_type_enum import ModelTypeEnum
from reinvent_models.model_factory.generative_model_base import GenerativeModelBase
from reinvent_models.model_factory.lib_invent_adapter import LibInventAdapter
from reinvent_models.model_factory.link_invent_adapter import LinkInventAdapter
from reinvent_models.model_factory.reinvent_core_adapter import ReinventCoreAdapter

class GenerativeModel:
    def __new__(cls, configuration: ModelConfiguration) -> GenerativeModelBase:
        cls._configuration = configuration
        model_type_enum = ModelTypeEnum()

        if cls._configuration.model_type == model_type_enum.DEFAULT:
            model = ReinventCoreAdapter(cls._configuration.model_file_path, mode=cls._configuration.model_mode)
        elif cls._configuration.model_type == model_type_enum.LIB_INVENT:
            model = LibInventAdapter(cls._configuration.model_file_path, mode=cls._configuration.model_mode)
        elif cls._configuration.model_type == model_type_enum.LINK_INVENT:
            model = LinkInventAdapter(cls._configuration.model_file_path, mode=cls._configuration.model_mode)
        else:
            raise ValueError(f"Invalid model_type provided: '{cls._configuration.model_type}")
        return model

