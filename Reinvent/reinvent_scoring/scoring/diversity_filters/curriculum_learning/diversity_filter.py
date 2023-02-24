from reinvent_scoring.scoring.diversity_filters.curriculum_learning import DiversityFilterParameters
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.base_diversity_filter import BaseDiversityFilter
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.identical_murcko_scaffold import \
    IdenticalMurckoScaffold
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.identical_topological_scaffold import \
    IdenticalTopologicalScaffold
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.no_filter import NoFilter
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.no_filter_with_penalty import NoFilterWithPenalty
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.scaffold_similarity import ScaffoldSimilarity


class DiversityFilter:

    def __new__(cls, parameters: DiversityFilterParameters) -> BaseDiversityFilter:
        all_filters = dict(IdenticalMurckoScaffold=IdenticalMurckoScaffold,
                           NoFilterWithPenalty=NoFilterWithPenalty,
                           IdenticalTopologicalScaffold=IdenticalTopologicalScaffold,
                           ScaffoldSimilarity=ScaffoldSimilarity,
                           NoFilter=NoFilter
                           )
        div_filter = all_filters.get(parameters.name, KeyError(f"Invalid filter name: `{parameters.name}'"))
        return div_filter(parameters)
