from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Lipinski import NumHAcceptors, NumHDonors
from rdkit import DataStructs
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt, CalcTPSA
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem.Lipinski import *
from rdkit.Chem.AtomPairs import Torsions, Pairs
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from tqdm import tqdm
from typing import List
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.rdMolDescriptors import CalcNumAtoms, CalcNumBridgeheadAtoms, CalcNumHBA, \
    CalcNumHBD, CalcNumAromaticRings, CalcNumAmideBonds, CalcNumRings, CalcNumSpiroAtoms, \
    CalcRadiusOfGyration, CalcLabuteASA
from sklearn.preprocessing import normalize
import pickle
import logging
from sklearn.metrics import roc_auc_score

import networkx as nx
from karateclub import Graph2Vec
import torch

def area_under_prc(pred, target):
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
    def __init__(self, smiles=None, labels=None):
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

    def dump(self, path):
        dict_fields = {'fps': self.fps, 'smiles': self.smiles, 'labels': self.labels, 'data_len': self.data_len}
        with open(path, 'wb+') as f:
            pickle.dump(dict_fields, f)

    def pick(self, path):
        with open(path, 'rb') as f:
            dict_fields = pickle.load(f)
        self.smiles = dict_fields['smiles']
        self.fps = dict_fields['fps']
        self.labels = dict_fields['labels']
        self.data_len = dict_fields['data_len']
        self.mols = [Chem.MolFromSmiles(s) for s in self.smiles]
        
    def split(self):
        lengths = [int(0.8 * len(self.data)), int(0.1 * len(self.data))]
        lengths += [len(self.data) - sum(lengths)]
        train_set, valid_set, test_set = torch.utils.data.random_split(self, lengths, generator=torch.Generator().manual_seed(1141))
        return train_set, valid_set, test_set
    
    def split(self):
        lengths = [int(0.8 * self.data_len), int(0.1 * self.data_len)]
        lengths += [self.data_len - sum(lengths)]
        train_ind, valid_ind, test_ind = torch.utils.data.random_split(self, lengths, generator=torch.Generator().manual_seed(1141))
        self.train_ind = train_ind
        self.valid_ind = valid_ind
        self.test_ind = test_ind

# data = pd.read_csv('../data/HIV.csv')
# data_fp = FPDataLoader(data.smiles.values, data.HIV_active.values)
# data_fp.prepare_fps()
# data_fp.dump('data/HIV_preprocessed')
# logging.warning('data dumped')
#
# data_fp = FPDataLoader()
# data_fp.pick('data/HIV_preprocessed')
# data_fp.fps = data_fp.fps[:, :-1]
# logging.warning('data loaded')
#
# X_train, X_test, y_train, y_test = train_test_split(data_fp.fps, data_fp.labels,
#                                                     test_size=0.3, random_state=1141)
#
# rf = RandomForestClassifier(random_state=1141, max_features='sqrt', n_estimators=500,
#                             min_samples_split=2, )
# param_grid = {'max_depth': [None], 'min_samples_split': [2],
#               'max_leaf_nodes': [None], 'n_estimators': [500]}
# # CV_search = GridSearchCV(estimator=rf,
# #                          param_grid=param_grid,
# #                          scoring='roc_auc', cv=3, verbose=2)
# rf.fit(X_train, y_train)
# logging.warning('model fitted')
# # print(CV_search.best_score_)
# # print(CV_search.best_estimator_)
# # print(CV_search.best_params_)
# y_pred = rf.predict(X_test)
# print(classification_report(y_test, y_pred))
# print('roc-auc', roc_auc_score(y_test, y_pred))
# print('prc-auc', area_under_prc(y_test, y_pred))
# # rf.fit(data_fp.fps, data_fp.labels)
# # print(accuracy_score(data_fp.labels, rf.predict(data_fp.fps)))
# print(0)
