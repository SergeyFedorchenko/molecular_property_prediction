from rdkit.Chem.Lipinski import *
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from typing import List, Union, Dict, Any, Optional, Set
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.rdMolDescriptors import CalcNumAtoms, CalcNumBridgeheadAtoms, CalcNumHBA, \
    CalcNumHBD, CalcNumAromaticRings, CalcNumAmideBonds, CalcNumRings, CalcNumSpiroAtoms, \
    CalcLabuteASA
from sklearn.preprocessing import normalize
import pickle
import torch


def area_under_prc(pred: npt.ArrayLike, target: npt.ArrayLike):
    """
    Area under precision-recall curve (PRC).

    Parameters:
        pred (np.array): predictions of shape (n,)
        target (np.array): binary targets of shape (n,)
    """
    order = np.argsort(pred)[::-1]
    target = target[order]
    precision = np.cumsum(target) / np.arange(1, len(target) + 1)
    auprc = precision[target == 1].sum() / ((target == 1).sum() + 1e-10)
    return auprc


class FPDataLoader:
    def __init__(self, smiles: Union[List[str], np.ndarray] = None,
                 labels: Union[List[int], np.ndarray] = None):
        self.smiles = smiles
        if smiles is not None:
            self.mols = np.array([Chem.MolFromSmiles(x) for x in smiles])
            self.data_len = len(smiles)
        else:
            self.mols = None
            self.data_len = None
        self.labels = labels

    def prepare_fps(self, fps_names: List[str] = ['morgan', 'rdk', 'mac'],
                    drop_corr: bool = True, add_decs: bool = True):
        fps_list = []
        if 'morgan' in fps_names:
            fps_list += [np.array(
                [AllChem.GetMorganFingerprintAsBitVect(x, radius=3, nBits=3 * 1024) for x
                 in tqdm(self.mols)], dtype=np.int8)]
        if 'rdk' in fps_names:
            rdkbi = {}
            fps_list += [np.array(
                [Chem.RDKFingerprint(x, maxPath=5, bitInfo=rdkbi) for x in
                 tqdm(self.mols)],
                dtype=np.int8)]
        if 'mac' in fps_names:
            fps_list += [np.array([MACCSkeys.GenMACCSKeys(x) for x in tqdm(self.mols)],
                                  dtype=np.int8)]

        joined_fps = np.concatenate(fps_list, axis=1)
        if drop_corr:
            corr = np.corrcoef(joined_fps, rowvar=False)
            corr[np.tril_indices(corr.shape[0])] = np.nan

            to_drop = np.array([any(abs(x) > 0.9) for x in corr])
            joined_fps = joined_fps[:, ~to_drop]
        if add_decs:
            for func in tqdm(
                    [CalcNumAtoms, CalcNumBridgeheadAtoms, CalcNumHBA, CalcNumHBD,
                     CalcNumAromaticRings, CalcNumAmideBonds, CalcNumRings,
                     CalcNumSpiroAtoms, CalcLabuteASA, ExactMolWt]):
                desc = normalize(np.array([func(m) for m in self.mols]).reshape(-1, 1),
                                 axis=0)
                joined_fps = np.concatenate([joined_fps, desc], axis=1)
        self.fps = joined_fps

    def dump(self, path: str):
        dict_fields = {'fps': self.fps, 'smiles': self.smiles, 'labels': self.labels,
                       'data_len': self.data_len}
        with open(path, 'wb+') as f:
            pickle.dump(dict_fields, f)

    def pick(self, path: str):
        with open(path, 'rb') as f:
            dict_fields = pickle.load(f)
        self.smiles = dict_fields['smiles']
        self.fps = dict_fields['fps']
        self.labels = dict_fields['labels']
        self.data_len = dict_fields['data_len']
        self.mols = [Chem.MolFromSmiles(s) for s in self.smiles]
