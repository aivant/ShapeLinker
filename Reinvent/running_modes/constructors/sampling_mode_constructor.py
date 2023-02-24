from dacite import from_dict

from running_modes.configurations.sampling.link_invent_sampling_configuration import LinkInventSamplingConfiguration
from running_modes.constructors.base_running_mode import BaseRunningMode
from running_modes.configurations import GeneralConfigurationEnvelope, SampleFromModelConfiguration
from running_modes.sampling.sample_from_model import SampleFromModelRunner
from running_modes.sampling.link_invent_sample_runner import SampleLinkInventModelRunner
from running_modes.utils.general import set_default_device_cuda
from running_modes.enums.model_type_enum import ModelTypeEnum


class SamplingModeConstructor:
    def __new__(self, configuration: GeneralConfigurationEnvelope) -> BaseRunningMode:
        self._configuration = configuration
        set_default_device_cuda()
        model_type_enum = ModelTypeEnum()

        if self._configuration.model_type == model_type_enum.DEFAULT:
            config = from_dict(data_class=SampleFromModelConfiguration, data=self._configuration.parameters)
            runner = SampleFromModelRunner(self._configuration, config)
        
        elif self._configuration.model_type == model_type_enum.LINK_INVENT:
            config = from_dict(data_class=LinkInventSamplingConfiguration, data=self._configuration.parameters)
            runner = SampleLinkInventModelRunner(self._configuration, config)
        
        elif self._configuration.model_type == model_type_enum.LIB_INVENT:
            raise NotImplementedError
        
        else:
            raise ValueError("Unknown model type: {}".format(self._configuration.model_type))
        
        return runner