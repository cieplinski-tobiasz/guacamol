"""
Code wrapping smina docking

Note: recommended way to use is to pass mol2 ligands, not pdbqt. Parsing files to pdbqt can be tricky and buggy.
Note: it is recommended to use a larger docking pocket. Otherwise many ligands fail to dock.
"""

# TODO: use guacamol-specific exceptions
# TODO: write test cases
import logging
import os
import subprocess
import tempfile

from rdkit import Chem
from rdkit.Chem import AllChem

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


def _embed_rdkit_molecule(molecule, seed: int, silent: bool = True):
    """
    Embeds RDKit molecule in place.

    The function tries to embed the molecule using random coordinates firstly.
    If this fails the molecule is embedded with not random coordinates.
    The molecule passed as an argument is left intact if both trials fail.

    Args:
        molecule: RDKit molecule
        seed (int): Seed used for embedding
        silent (bool): If False, exception is thrown when embedding fails.
                          Otherwise, None is returned.

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
        logger.warning('Embedding failure')

        if silent:
            return None
        else:
            # TODO: SCBC Excpetion
            raise RuntimeError(f'Embedding failure')

    return molecule


def _optimize_rdkit_molecule(molecule, silent: bool = False):
    """
    Optimizes the structure of RDKit molecule in place.

    The function tries a number of maxIters parameters.
    The molecule passed as an argument is left intact
    if none of the optimization trials succeed.

    Args:
        molecule: RDKit molecule
        silent (bool): If False, exception is thrown when optimization fails.
                          Otherwise, None is returned.

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
        logger.warning('Structure optimization failure')

        if silent:
            return None
        else:
            # TODO: SCBC Excpetion
            raise RuntimeError('Structure optimization failure')

    return molecule


def _to_mol2_file(smiles: str, output_filename: str, seed: int = 0, silent: bool = False):
    """
    Converts a SMILES string to mol2 file.

    Args:
        smiles (str): SMILES string of the molecule
        output_filename (str): Path of the file the converted molecule will be saved to
        seed (int): Seed used during optimizing and embedding molecule
        silent (bool): If False, exception is thrown when conversion fails.
                          Otherwise, None is returned.

    Returns:
        Output filename if the conversion succeeds, else None if silent is True.
        The returned filename is the same filename that was passed as an argument.

    Raises:
        RuntimeError: If conversion fails and silent is False.
    """
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        # TODO: InvalidMolecule
        raise RuntimeError(f'RDKit conversion failure for {smiles}')

    molecule = _embed_rdkit_molecule(molecule, seed)
    _optimize_rdkit_molecule(molecule)
    Chem.MolToMolFile(molecule, output_filename)

    # TODO: either change to subprocess.run
    #       or mute stdout/stderr somehow
    openbabel_return_code = os.system(f'obabel -imol {output_filename} -omol2 -O {output_filename}')

    if openbabel_return_code != 0:
        logger.warning(f'Mol2 conversion for {smiles} failed')

        if silent:
            return None
        else:
            # TODO: InvalidMolecule?
            raise RuntimeError(f'Openbabel conversion failure for {smiles}')

    return output_filename


def _exec_command(command: str, timeout: int = None):
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, timeout=timeout)
        out, err, return_code = str(result.stdout, 'utf-8').split('\n'), str(result.stderr, 'utf-8').split(
            '\n'), result.returncode
        return out, err, return_code
    except subprocess.TimeoutExpired: # TODO: raise ScoreCannotBeCalculated
        return [], [], 'timeout'
    except Exception as e:
        raise e


def _parse_smina_result(docked_ligand_fname, smina_stdout):
    assert docked_ligand_fname.endswith('mol2')

    # WARNING: A Very hacky way to get score table
    results_start_index = smina_stdout.index('-----+------------+----------+----------')
    score_table = smina_stdout[results_start_index + 1:-3]
    scores = [float(row.split()[1]) for row in score_table]

    return scores


def smina_dock_ligand(smiles, receptor, pocket_center=None, pocket_range=25, exhaustiveness=16, seed=0,
                      timeout=600, n_cpu=8):
    """
    Docks the passed molecule

    Args:
        smiles (str): Ligand in SMILES format
        receptor (str): Path to receptor in pdbqt or mol2 format
        pocket_center (numpy.arrray): x, y, z coordinates of docking center. Must be of shape (3, ).
        pocket_range (list, int): How far from the center are we trying to dock. If list is passed it must be of size 3.
        exhaustiveness (int): Best practice is to bump to 16 (as compared to Vina 8s)
        seed (int): Random seed passed to the smina simulator
        timeout (int): Maximum waiting time in seconds
        n_cpu (int): How many cpus to use

    Returns:
        float: Docking score of the ligand
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
        stdout, stderr, return_code = _exec_command(cmd, timeout=timeout)

        if return_code != 0:
            logger.error(stderr)
            # TODO: ScoreCannotBeCalculated
            raise RuntimeError(f'Failed to dock {os.path.basename(smiles)} to {os.path.basename(receptor)}')

        scores = _parse_smina_result(output.name, stdout)

        return min(scores)
