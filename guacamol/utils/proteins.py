import inspect
import os
import json

import guacamol.data

BUILTIN_PROTEINS_DIR = os.path.join(os.path.dirname(inspect.getfile(guacamol.data)), 'proteins')
PROTEINS_ENV_VAR = 'GUACAMOL_PROTEINS'


class Protein:
    _METADATA_FILENAME = 'metadata.json'

    def __init__(self, directory):
        self._directory = directory

    @property
    def path(self) -> os.PathLike:
        # TODO: cache
        protein_files = [entry for entry in os.listdir(self._directory)
                         if entry.endswith('.pdb') or entry.endswith('.pdbqt')]

        if len(protein_files) != 1:
            raise RuntimeError('Ambiguous files inside the protein directory')

        return os.path.join(self._directory, protein_files[0])

    @property
    def metadata(self) -> dict:
        with open(os.path.join(self._directory, self.__class__._METADATA_FILENAME)) as metadata_file:
            return json.load(metadata_file)

    @property
    def pocket_center(self):
        return self.metadata['pocket_center']


def get_proteins():
    return {protein_dir.lower(): Protein(os.path.join(BUILTIN_PROTEINS_DIR, protein_dir))
            for protein_dir in os.listdir(BUILTIN_PROTEINS_DIR)}
