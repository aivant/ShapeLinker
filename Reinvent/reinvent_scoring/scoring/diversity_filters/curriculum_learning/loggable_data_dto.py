from dataclasses import dataclass


@dataclass
class UpdateLoggableDataDTO:
    """This class is used by the Diversity Filters to log out metadata."""
    input: str
    output: str
    likelihood: float = None
    prior_likelihood: float = None
