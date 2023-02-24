from dataclasses import dataclass


@dataclass(frozen=True)
class ROCSSimilarityMeasuresEnum():
    TANIMOTO = "Tanimoto"
    REF_TVERSKY = "RefTversky"
    FIT_TVERSKY = "FitTversky"
