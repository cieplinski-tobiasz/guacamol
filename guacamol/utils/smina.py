"""
Code wrapping smina docking

Note: recommended way to use is to pass mol2 ligands, not pdbqt. Parsing files to pdbqt can be tricky and buggy.
Note: it is recommended to use a larger docking pocket. Otherwise many ligands fail to dock.
"""

# TODO: write test cases
import logging
import os
import subprocess
import tempfile
from typing import Union, List, Tuple, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

import guacamol.utils.chemistry as chem_utils
from guacamol.scoring_function import InvalidMolecule

logger = logging.getLogger(__name__)

proteins = {
    '5ht1b': {
        'filename': '5HT1B_4IAQ.pdbqt',
        'pocket_center': (-26.602, 5.277, 17.898),
    },
    '5ht2b': {
        'filename': '5HT2B_4NC3.pdbqt',
        'pocket_center': (-16.210, -15.874, 5.523),
    },
    'acm2': {
        'filename': '5HT2B_4NC3.pdbqt',
        'pocket_center': (18.847, -3.093, -2.861),
    },
}


def _embed_rdkit_molecule(molecule: Chem.Mol, seed: int, silent: bool = True) -> Optional[Chem.Mol]:
    """
    Embeds RDKit molecule in place.

    The function tries to embed the molecule using random coordinates firstly.
    If this fails the molecule is embedded with not random coordinates.
    The molecule passed as an argument is left intact if both trials fail.

    Args:
        molecule: RDKit molecule
        seed: Seed used for embedding
        silent: If False, exception is thrown when embedding fails. Otherwise, None is returned.

    Returns:
        Embedded RDKit molecule if embedding succeeds.
        Otherwise, if silent is True, returns None.
        The returned molecule is the same molecule that was passed as an argument.

    Raises:
        RuntimeError: If embedding fails and silent is False.
    """
    molecule = Chem.AddHs(molecule)
    conf_id = AllChem.EmbedMolecule(molecule, useRandomCoords=True, ignoreSmoothingFailures=True, randomSeed=seed)

    if conf_id == -1:
        conf_id = AllChem.EmbedMolecule(molecule, useRandomCoords=False, ignoreSmoothingFailures=True, randomSeed=seed)

    if conf_id == -1:
        if silent:
            return None
        else:
            raise InvalidMolecule(f'Embedding failure')

    return molecule


def _optimize_rdkit_molecule(molecule: Chem.Mol, silent: bool = False) -> Optional[Chem.Mol]:
    """
    Optimizes the structure of RDKit molecule in place.

    The function tries a number of maxIters parameters.
    The molecule passed as an argument is left intact
    if none of the optimization trials succeed.

    Args:
        molecule: RDKit molecule
        silent: If False, exception is thrown when optimization fails. Otherwise, None is returned.

    Returns:
        Embedded RDKit molecule if optimization succeeds.
        Otherwise, if silent is True, returns None.
        The returned molecule is the same molecule that was passed as an argument.

    Raises:
        RuntimeError: If optimization fails and silent is False.
    """
    for max_iterations in [200, 2000, 20000, 200000]:
        ret = AllChem.UFFOptimizeMolecule(molecule, maxIters=max_iterations)

        if ret == 0:
            break

    if ret != 0:
        if silent:
            return None
        else:
            raise InvalidMolecule('Structure optimization failure')

    return molecule


def _to_mol2_file(smiles: str, output_filename: str, seed: int = 0, silent: bool = False) -> Optional[str]:
    """
    Converts a SMILES string to mol2 file.

    Args:
        smiles: SMILES string of the molecule
        output_filename: Path of the file the converted molecule will be saved to
        seed: Seed used during optimizing and embedding molecule
        silent: If False, exception is thrown when conversion fails. Otherwise, None is returned.

    Returns:
        Output filename if the conversion succeeds, else None if silent is True.
        The returned filename is the same filename that was passed as an argument.

    Raises:
        RuntimeError: If conversion fails and silent is False.
    """
    molecule = chem_utils.smiles_to_rdkit_mol(smiles)

    if molecule is None:
        raise InvalidMolecule(f'Failed to convert {smiles} to RDKit mol')

    molecule = _embed_rdkit_molecule(molecule, seed)
    _optimize_rdkit_molecule(molecule)
    Chem.MolToMolFile(molecule, output_filename)

    command = f'obabel -imol {output_filename} -omol2 -O {output_filename}'
    openbabel_return_code = subprocess.run(command, shell=True, stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL).returncode

    if openbabel_return_code != 0:
        if silent:
            return None
        else:
            raise InvalidMolecule(f'Failed to convert {smiles} to .mol2')

    return output_filename


def _exec_docking(command: str, timeout: int = None) -> Tuple[List[str], List[str], int]:
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, timeout=timeout)
        out, err, return_code = str(result.stdout, 'utf-8').split('\n'), str(result.stderr, 'utf-8').split(
            '\n'), result.returncode
        return out, err, return_code
    except subprocess.TimeoutExpired:
        raise InvalidMolecule('Docking timeout')


def _parse_smina_result(docked_ligand_fname: str, smina_stdout: List[str]) -> List[float]:
    assert docked_ligand_fname.endswith('mol2')

    # WARNING: A Very hacky way to get score table
    results_start_index = smina_stdout.index('-----+------------+----------+----------')
    score_table = smina_stdout[results_start_index + 1:-3]
    scores = [float(row.split()[1]) for row in score_table]

    return scores


def smina_dock_ligand(smiles: str, receptor: str, pocket_center: Union[List, np.array, Tuple] = None,
                      pocket_range: Union[int, List[int], np.array, Tuple[int]] = 25, exhaustiveness: int = 16,
                      seed: int = 0, timeout: int = 600, n_cpu: int = 8) -> float:
    """
    Docks the passed molecule

    Args:
        smiles: Ligand in SMILES format
        receptor: Path to receptor in pdbqt or mol2 format
        pocket_center: x, y, z coordinates of docking center
        pocket_range: How far from the center are we trying to dock. If list is passed it must be of size 3.
        exhaustiveness: Best practice is to bump to 16 (as compared to Vina 8s)
        seed: Random seed passed to the smina simulator
        timeout: Maximum waiting time in seconds
        n_cpu: How many cpus to use

    Returns:
        Docking score of the ligand
    """
    if type(pocket_range) is int:
        pocket_range = [pocket_range] * 3

    with tempfile.NamedTemporaryFile(suffix='.mol2') as ligand, tempfile.NamedTemporaryFile(suffix='.mol2') as output:
        _to_mol2_file(smiles, ligand.name)

        cmd = [
            'smina',
            '--receptor', receptor,
            '--ligand', ligand.name,
            '--seed', seed,
            '--cpu', n_cpu,
            '--out', output.name,
            '--center_x', pocket_center[0],
            '--center_y', pocket_center[1],
            '--center_z', pocket_center[2],
            '--size_x', pocket_range[0],
            '--size_y', pocket_range[1],
            '--size_z', pocket_range[2],
            '--exhaustiveness', exhaustiveness,
        ]

        cmd = ' '.join([str(entry) for entry in cmd])
        stdout, stderr, return_code = _exec_docking(cmd, timeout=timeout)

        if return_code != 0:
            logger.error(stderr)
            raise InvalidMolecule(f'Failed to dock {smiles} to {os.path.basename(receptor)}')

        scores = _parse_smina_result(output.name, stdout)

        return min(scores)
