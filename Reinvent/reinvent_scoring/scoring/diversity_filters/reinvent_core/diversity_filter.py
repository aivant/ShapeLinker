from reinvent_scoring.scoring.diversity_filters.reinvent_core import IdenticalMurckoScaffold, \
    IdenticalTopologicalScaffold, ScaffoldSimilarity, NoScaffoldFilter
from reinvent_scoring.scoring.diversity_filters.reinvent_core.base_diversity_filter import BaseDiversityFilter
from reinvent_scoring.scoring.diversity_filters.reinvent_core.diversity_filter_parameters import \
    DiversityFilterParameters


class DiversityFilter:

    def __new__(cls, parameters: DiversityFilterParameters) -> BaseDiversityFilter:
        all_filters = dict(IdenticalMurckoScaffold=IdenticalMurckoScaffold,
                           IdenticalTopologicalScaffold=IdenticalTopologicalScaffold,
                           ScaffoldSimilarity=ScaffoldSimilarity,
                           NoFilter=NoScaffoldFilter)
        div_filter = all_filters.get(parameters.name, KeyError(f"Invalid filter name: `{parameters.name}'"))
        return div_filter(parameters)
