from collections import namedtuple

from openeye import oeshape

SIM_FUNC = namedtuple('SIM_FUNC', ['shape', 'color', 'predicate'])

def get_similarity_name_set(parameters, param_names_enum, sim_measure_enum):
    similarity_collection_name = parameters.specific_parameters.get(param_names_enum.SIM_MEASURE,
                                                                         sim_measure_enum.TANIMOTO)
    return similarity_collection(similarity_collection_name, sim_measure_enum)


def similarity_collection(sim_measure_type, sim_measure_enum):
    sim_def_dict = {
        sim_measure_enum.TANIMOTO: SIM_FUNC('GetTanimoto', 'GetColorTanimoto', 'OEHighestTanimotoCombo'),
        sim_measure_enum.REF_TVERSKY: SIM_FUNC('GetRefTversky', 'GetRefColorTversky',
                                                     'OEHighestRefTverskyCombo'),
        sim_measure_enum.FIT_TVERSKY: SIM_FUNC('GetFitTversky', 'GetFitColorTversky',
                                                     'OEHighestFitTverskyCombo'),
    }
    return sim_def_dict.get(sim_measure_type)

def init_cff(prep, cff_path):
    if len(cff_path) == 0:
        cff_path = oeshape.OEColorFFType_ImplicitMillsDean
    cff = oeshape.OEColorForceField()
    if cff.Init(cff_path):
        prep.SetColorForceField(cff)
    else:
        raise ValueError("Custom color force field initialisation failed")