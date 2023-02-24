from typing import Tuple, Dict

from pymol import cmd
import numpy as np
from scipy import spatial
import pandas as pd
from openbabel import openbabel
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.rdFMCS import FindMCS


def lig_protein_clash_dist(poi_path: str, protac_path: str, cutoff: float = 1.2, return_mindist: bool = False) -> Tuple[float, int]:
    """
    Calculate the number of protac atoms which are less than cutoff A away from the protein.
    :param poi_path: Path to the protein of interest.
    :param protac_path: Path to the protac.
    :param cutoff: Cutoff distance in Angstroms.
    :return: Tuple of the minimum distance and number of clashes
    """
    cmd.reinitialize()
    cmd.load(poi_path, 'poi')
    cmd.load(protac_path, 'protac')
    
    ptc_coords = cmd.get_coords(selection='protac')
    ref_coords = cmd.get_coords(selection=f'poi')

    dist_mat = spatial.distance_matrix(
        np.array(ptc_coords), np.array(ref_coords)
    )
    ref_clash = []
    for ptc_atom, dists in enumerate(dist_mat):
        n_prot_clash = len([d for d in dists if d <= cutoff])
        ref_clash.append({
            'protac_atom': ptc_atom,
            'n_protein_clashes': n_prot_clash,
            'min_dist': dists.min()
        })
    ref_clash = pd.DataFrame(ref_clash)
    n_clash = ref_clash.n_protein_clashes.sum()
    min_dist = ref_clash.min_dist.min()
    if return_mindist:
        return min_dist, n_clash
    else:
        return n_clash
    

def lig_protein_clash_vdw(poi_path: str, protac_path: str, return_mindist: bool = False) -> Tuple[float, int]:
    """
    Calculate the number of clashes between the protac and the protein.
    A clash is defined as any protac atom which is closer to a protein atom than the sum of their vdw radii.
    :param poi_path: Path to the protein of interest.
    :param protac_path: Path to the protac.
    :return: Tuple of the minimum distance and number of clashes.
    """
    cmd.reinitialize()
    cmd.load(poi_path, 'poi')
    cmd.load(protac_path, 'protac')
    cmd.remove('hydrogens')
    
    ptc_coords = cmd.get_coords(selection='protac')
    ref_coords = cmd.get_coords(selection=f'poi')
    vdw_radii = {'ref_vdw_radii': [],
             'poi_vdw_radii': []}
    cmd.iterate('protac', 'ref_vdw_radii.append(vdw)', quiet=1, space=vdw_radii)
    cmd.iterate('poi', 'poi_vdw_radii.append(vdw)', quiet=1, space=vdw_radii)

    dist_mat = spatial.distance_matrix(
        np.array(ptc_coords), np.array(ref_coords)
    )
    ref_clash = []
    for ptc_atom, dists in enumerate(dist_mat):
        ptc_vdw = vdw_radii['ref_vdw_radii'][ptc_atom]
        combined_radii = np.array(vdw_radii['poi_vdw_radii']) + ptc_vdw
        # clash if distance is less than sum of vdw radii
        n_prot_clash = 0
        for i, d in enumerate(dists):
            if d <= combined_radii[i]:
                n_prot_clash += 1
        ref_clash.append({
            'protac_atom': ptc_atom,
            'n_protein_clashes': n_prot_clash,
            'min_dist': dists.min()
        })
    ref_clash = pd.DataFrame(ref_clash)
    n_clash = ref_clash.n_protein_clashes.sum()
    min_dist = ref_clash.min_dist.min()
    if return_mindist:
        return min_dist, n_clash
    else:
        return n_clash
    

def calc_torsion_energy(sdf_file: str,
                force_field: str = 'MMFF94',
                addH: bool = False) -> Dict[str, float]:
    # Obabel ligand energy
    # Read the file.
    mol = openbabel.OBMol()
    conv = openbabel.OBConversion()
    format = conv.FormatFromExt(sdf_file)
    conv.SetInAndOutFormats(format, format)
    conv.ReadFile(mol, sdf_file)
    if addH:
        mol.AddHydrogens()
    # Find the MMFF94 force field.
    ff = openbabel.OBForceField.FindForceField(force_field)
    if ff == 0:
        print("Could not find forcefield")

    # Set the log level to high since we want to print out individual
    # interactions.
    ff.SetLogLevel(openbabel.OBFF_LOGLVL_NONE)
    # python specific, python doesn't have std::ostream so the SetLogFile()
    # function is replaced by SetLogToStdOut and SetLogToStdErr in the SWIG
    # interface file
    ff.SetLogToStdErr()
    # Setup the molecule. This assigns atoms types, charges and parameters
    if ff.Setup(mol) == 0:
        print("Could not setup forcefield")
    # Calculate the energy
    return ff.E_Torsion()


def mcs_rmsd(probe_mol: Mol, ref_mol: Mol) -> float:
    """Calculate molecule RMSD based on best MCS match.
    Parameters
    ----------
    probe_mol : Mol
        Probe molecule to compute RMSD with respect to `ref_mol`.
    ref_mol : Mol
        Reference molecule to use for MCS matching.
    Returns
    -------
    float
        Smallest RMSD possible based on enumerated MCS atom maps.
    """
    mcs = FindMCS(
        [probe_mol, ref_mol],
        completeRingsOnly=False,
        ringMatchesRingOnly=False,
        timeout=1
    )
    patt = Chem.MolFromSmarts(mcs.smartsString)
    atom_maps_probe = np.array(
        probe_mol.GetSubstructMatches(
            patt, uniquify=False, maxMatches=250
        )
    )
    atom_maps_ref = np.array(
        ref_mol.GetSubstructMatches(
            patt, uniquify=False, maxMatches=250
        )
    )
    # Compute RMSD
    xyz_ref = ref_mol.GetConformer(0).GetPositions()
    xyz_probe = probe_mol.GetConformer(0).GetPositions()

    # Read conformer
    results = []
    for map_probe in atom_maps_probe:
        for map_ref in atom_maps_ref:
            map_lig = [(i, j) for i, j in zip(map_probe, map_ref)]
            match_ref = xyz_probe[[at[0] for at in map_lig]]
            ref_coord = xyz_ref[[at[1] for at in map_lig]]
            rms = np.sqrt(
                np.square(
                    np.linalg.norm(match_ref - ref_coord, axis=1)
                ).mean()
            )
            results.append([rms, map_probe, map_lig])
    results = pd.DataFrame(
        results,
        columns=['rmsd', 'probe_atom_map', 'ref_atom_map']
    ).sort_values(by='rmsd').reset_index(drop=True)
    return results.rmsd.min()