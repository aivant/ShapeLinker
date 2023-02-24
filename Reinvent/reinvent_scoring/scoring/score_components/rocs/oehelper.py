from openeye import oechem, oeshape, oeomega
from rdkit import Chem

def get_omega_confs(imol, omega, enum_stereo, max_stereo):
    stereo = False
    no_stereo = False
    if enum_stereo:
        enantiomers = list(oeomega.OEFlipper(imol.GetActive(), max_stereo, False, True))
        for k, enantiomer in enumerate(enantiomers):
            # Any other simpler way to combine and add all conformers to imol have failed !!
            # Failure = Creates conformers with wrong indices and wrong connections
            enantiomer = oechem.OEMol(enantiomer)
            ret_code = omega.Build(enantiomer)
            if ret_code == oeomega.OEOmegaReturnCode_Success:
                if k == 0:
                    imol = oechem.OEMol(enantiomer.SCMol())
                    imol.DeleteConfs()
                stereo = True
                for x in enantiomer.GetConfs():
                    imol.NewConf(x)
    else:
        no_stereo = omega(imol)
    return no_stereo or stereo, imol

def get_score(mol, score, sim_func_name_set, shape_weight, color_weight, neg_prot_file, neg_lig_file):
    neg_score = 0.0
    if len(neg_prot_file) > 0:
        neg_score = neg_vol_score(mol, neg_prot_file, neg_lig_file)

    best_score_shape = getattr(score, sim_func_name_set.shape)()
    best_score_shape = correct_shape_score(best_score_shape)
    best_score_shape = penalise_neg_volume(best_score_shape, neg_score)

    best_score_color = getattr(score, sim_func_name_set.color)()
    best_score_color = correct_color_score(best_score_color)
    best_score = ((shape_weight * best_score_shape) + (
            color_weight * best_score_color)) / (shape_weight + color_weight)
    return best_score, best_score_shape, best_score_color, neg_score

def neg_vol_score(mol, neg_prot_file, neg_lig_file):
    # 'mol' is the active conformation as obtained from overlay with the main query
    # It is important to use the protein ligand and do the overlay with mol again (query is the protein)
    # otherwise the score calculation is incorrect
    qfs = oechem.oemolistream()
    if not qfs.open(neg_lig_file):
        raise ValueError(f'Ligand file {neg_lig_file} could not be opened')
    qmol = oechem.OEMol()
    oechem.OEReadMolecule(qfs, qmol)

    efs = oechem.oemolistream()
    if not efs.open(neg_prot_file):
        raise ValueError(f'Protein file {neg_prot_file} could not be opened')
    emol = oechem.OEMol()
    oechem.OEReadMolecule(efs, emol)

    res = oeshape.OEROCSResult()
    evol = oeshape.OEExactShapeFunc()
    evol.SetupRef(emol)

    oeshape.OEROCSOverlay(res, qmol, mol)
    outmol = res.GetOverlayConf()

    # calculate overlap with protein
    eres = oeshape.OEOverlapResults()
    evol.Overlap(outmol, eres)

    frac = eres.GetOverlap() / eres.GetFitSelfOverlap()
    return frac

def prep_sdf_file(outmol, score, smile, batch_id, best_score_shape, best_score_color, neg_score):
    mol = Chem.MolFromSmiles(smile)
    smile = Chem.MolToSmiles(mol, canonical=True) if mol else ""
    score.Transform(outmol)
    oechem.OESetSDData(outmol, "Batch ID", batch_id)
    oechem.OESetSDData(outmol, "Smiles", smile)
    oechem.OESetSDData(outmol, "Shape", "%-.3f" % best_score_shape)
    oechem.OESetSDData(outmol, "Color", "%-.3f" % best_score_color)
    oechem.OESetSDData(outmol, "Negative vol penalty", "%-.3f" % neg_score)

def correct_color_score(score):
    if score >= 1.0:
        score = 0.90  # or alternative
    return score

def correct_shape_score(score):
    if score >= 1.0:
        score = 0.95  # or alternative
    return score

def penalise_neg_volume(score, neg_score):
    # Generally neg_score is not very high rouhly representing the % of molecule clashing with the protein
    # Assuming that even a low value is unacceptable, a different penalty function should be considered
    return score - neg_score