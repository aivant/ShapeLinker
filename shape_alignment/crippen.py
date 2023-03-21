import argparse
import multiprocessing
import os
import subprocess
from itertools import repeat
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, PandasTools, rdMolAlign, rdMolDescriptors
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMoleculeConfs
from rdkit.Chem.rdShapeHelpers import ShapeProtrudeDist, ShapeTanimotoDist
from sklearn.feature_extraction.text import CountVectorizer


def p2p_alignment(mol_probe, mol_ref, phar_contrib_probe, phar_contrib_ref):
    crippenO3A = GetCrippenO3A(
        mol_probe, mol_ref, phar_contrib_probe, phar_contrib_ref, 0, 0
    )
    crippenO3A.Align()
    rmsd = crippenO3A.Trans()[0]
    trans_mat = crippenO3A.Trans()[1]
    score = crippenO3A.Score()

    return [score, rmsd, trans_mat]



def how_many_conformers(mol):
    nb_rot_bonds = AllChem.CalcNumRotatableBonds(mol)
    if nb_rot_bonds <= 7:
        return 50
    elif nb_rot_bonds <= 12:
        return 200
    else:
        return 300


def GetChargeContribs(mol):
    mol.ComputeGasteigerCharges()
    gast = [at.GetDoubleProp("_GasteigerCharge") for at in mol.GetAtoms()]
    return [
(i, j[0], k)
for i, j, k in zip(
rdMolDescriptors.CalcEEMcharges(mol),
rdMolDescriptors._CalcCrippenContribs(mol),
gast,
)
]


class CrippenAlignment:
    def __init__(
            self,
            ref_mol: Mol,
            probe_mol_list: np.ndarray,
            probe_mol_names: np.ndarray,
            rms_thresh: float = 0.25,
    ) -> None:
        self.ref_mol = ref_mol
        self.probe_mol_list = [self.add_h(m) for m in probe_mol_list if m is not None]
        self.probe_mol_names = [
            n for i, n in enumerate(probe_mol_names) if probe_mol_list[i] is not None
        ]
        self.nconf_list = [how_many_conformers(m) for m in self.probe_mol_list]
        self.rms_thresh = rms_thresh
        self.factory = Gobbi_Pharm2D.factory
        self.ref_ph_fp = self.calculate_pcp_fp(self.ref_mol, 0)

    def calculate_pcp_fp(self, mol: Mol, cid: int):
        ph_fp = Generate.Gen2DFingerprint(
            mol, self.factory, dMat=Chem.Get3DDistanceMatrix(mol, confId=cid)
        )
        return ph_fp

    def pharmacophore_similarity(self, fp1, fp2):
        tani = DataStructs.TanimotoSimilarity(fp1, fp2)
        return tani

    def calculate_shape_metrics(
            self, ref_mol: Mol, probe_mol: Mol, ref_cid: int = 0, probe_cid: int = 0
    ) -> Tuple[float, float]:
        sh_psc = ShapeProtrudeDist(probe_mol, ref_mol, probe_cid, ref_cid)
        sh_tsc = 1.0 - ShapeTanimotoDist(probe_mol, ref_mol, probe_cid, ref_cid)
        return (sh_psc, sh_tsc)

    def add_h(self, mol: Mol) -> Mol:
        mol = Chem.AddHs(mol)
        return mol

    def embed_conformers(self, mol: Mol, nconf: int) -> None:
        if hasattr(AllChem, "ETKDGv3"):
            p = AllChem.ETKDGv3()
        elif hasattr(AllChem, "ETKDGv2"):
            p = AllChem.ETKDGv2()
        else:
            p = AllChem.ETKDG()
        p.pruneRmsThresh = self.rms_thresh
        AllChem.EmbedMultipleConfs(mol, nconf, p)
        MMFFOptimizeMoleculeConfs(mol, numThreads=0)

    def calculate_crippen_contribs(self, mol: Mol):
        return rdMolDescriptors._CalcCrippenContribs(mol)

    def crippen_align_molecules(
            self, probe_mol: Mol, crippen_prob_contrib: np.ndarray, nconf: int
    ) -> Union[Tuple[pd.DataFrame, Mol], None]:
        tempscore = []
        shapescores = []
        for cid in range(nconf):
            try:
                crippenO3A = rdMolAlign.GetCrippenO3A(
                    probe_mol,
                    self.ref_mol,
                    crippen_prob_contrib,
                    self.ref_contribs,
                    cid,
                    0,
                )
                crippenO3A.Align()
                score = crippenO3A.Score()
                tempscore.append(score)
                shape_metric = self.calculate_shape_metrics(
                    self.ref_mol, probe_mol, ref_cid=0, probe_cid=cid
                )
                shapescores.append(shape_metric)
            except ValueError:
                continue
            except TypeError as e:
                print(
                    f'Encountered TypeError during alignment: {str(e)}'
                )
        num_hac = probe_mol.GetNumHeavyAtoms()
        if shapescores:
            metrics_df = pd.DataFrame(shapescores)
            metrics_df.columns = [
                "Shape_Protrude_Distance",
                "Shape_Tanimoto_Similarity",
            ]
            metrics_df["Crippen_O3A_Score"] = [t / num_hac for t in tempscore]
            metrics_df["ShapeSim_Crippen_Combo"] = (
                metrics_df.Crippen_O3A_Score * metrics_df.Shape_Tanimoto_Similarity
            )
            # metrics_df = metrics_df.sort_values('ShapeSim_Crippen_Combo')
            metrics_df = metrics_df.sort_values("Crippen_O3A_Score")
            return metrics_df, probe_mol

    def do_alignment(
            self,
            method: str = "C",
            consensus_pharm: bool = False,
            out_file_name: str = "out.sdf",
    ) -> None:
        for mol, nconf in zip(self.probe_mol_list, self.nconf_list):
            try:
                self.embed_conformers(mol, nconf)
            except Exception as e:
                print(f"Failed to embed mol: {str(e)}")

        if method == "C":
            self.prob_contribs = [
                self.calculate_crippen_contribs(m) for m in self.probe_mol_list
            ]
            self.ref_contribs = self.calculate_crippen_contribs(self.ref_mol)
        elif method == "E":
            self.prob_contribs = [GetChargeContribs(m) for m in self.probe_mol_list]
            self.ref_contribs = GetChargeContribs(self.ref_mol)
        elif method == "P":
            if consensus_pharm:
                self.ref_fm = featmap_from_mol(self.ref_mol)
                self.ref_contribs = get_pharm_contrib(self.ref_fm, vectorizer)
                self.prob_contribs = [
                    get_pharm_contrib(featmap_from_mol(m), vectorizer)
                    for m in self.probe_mol_list
                ]
            else:
                self.ref_contribs = GetPharmContribs(self.ref_mol)
                self.prob_contribs = [GetPharmContribs(m) for m in self.probe_mol_list]

        w = Chem.SDWriter(out_file_name)
        for i, (pm, pc, nc) in enumerate(
            zip(self.probe_mol_list, self.prob_contribs, self.nconf_list)
        ):
            mol_name = self.probe_mol_names[i]
            aln = self.crippen_align_molecules(pm, pc, nc)
            if not isinstance(aln, tuple):
                continue
            df, aligned_mol = aln
            best_ids = df.index.values[-30:]
            for best in best_ids:
                aligned_mol.SetProp("Best Conformer", "%d" % best)
                aligned_mol.SetProp(
                    "Crippen Overlap Score",
                    "%3.2f" % df.Crippen_O3A_Score.values[best],
                )
                aligned_mol.SetProp(
                    "Shape Similarity Score",
                    "%3.2f" % df.Shape_Tanimoto_Similarity.values[best],
                )
                aligned_mol.SetProp(
                    "Crippen_and_Shape Combination Score",
                    "%3.2f" % df.ShapeSim_Crippen_Combo.values[best],
                )
                aligned_mol.SetProp("_Name", mol_name)
                w.write(aligned_mol, confId=int(best))
        w.close()
        final_df = PandasTools.LoadSDF(out_file_name, removeHs=False)
        final_df["Crippen_and_Shape Combination Score"] = final_df[
        "Crippen_and_Shape Combination Score"
        ].astype(float)
        final_df = final_df.sort_values(
            "Crippen_and_Shape Combination Score", ascending=False
        )
        PandasTools.WriteSDF(final_df, out_file_name, properties=final_df.columns)