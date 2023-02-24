import numpy as np
from rdkit.Chem import DataStructs


class Similarity:

    def calculate_tanimoto(self, query_fps, ref_fingerprints) -> np.array:
        return np.array([np.max(DataStructs.BulkTanimotoSimilarity(fp, ref_fingerprints)) for fp in query_fps])

    def calculate_jaccard_distance(self, query_fps, ref_fingerprints) -> np.array:
        tanimoto = self.calculate_tanimoto(query_fps, ref_fingerprints)
        jaccard = 1 - tanimoto
        return jaccard
