from reinvent_scoring.scoring.diversity_filters.lib_invent import NoFilter, NoFilterWithPenalty, IdenticalMurckoScaffold
from reinvent_scoring.scoring.diversity_filters.lib_invent.base_diversity_filter import BaseDiversityFilter
from reinvent_scoring.scoring.diversity_filters.lib_invent.diversity_filter_parameters import DiversityFilterParameters


class DiversityFilter:

    def __new__(cls, parameters: DiversityFilterParameters) -> BaseDiversityFilter:
        all_filters = dict(NoFilter=NoFilter,
                           IdenticalMurckoScaffold=IdenticalMurckoScaffold,
                           NoFilterWithPenalty=NoFilterWithPenalty)
        div_filter = all_filters.get(parameters.name, KeyError(f"Invalid filter name: `{parameters.name}'"))
        return div_filter(parameters)
