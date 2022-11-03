import os

from torchdrug import data, utils
from torchdrug.core import Registry as R
from torchdrug import core, models, tasks, datasets
import torch
from torch import nn, optim
from collections import defaultdict
import numpy as np

from torch.utils import data as torch_data

import logging
import json


@R.register("datasets.HIV_mols")
@utils.copy_args(data.MoleculeDataset.load_csv, ignore=("smiles_field", "target_fields"))
class HIV_mols(data.MoleculeDataset):
    """
    Qualitative data of drugs approved by the FDA and those that have failed clinical
    trials for toxicity reasons.
    Statistics:
        - #Molecule: 1,478
        - #Classification task: 2
    Parameters:
        path (str): path to store the dataset
        verbose (int, optional): output verbose level
        **kwargs
    """

    target_fields = ["HIV_active"]

    def __init__(self, path, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        csv_file = path

        self.load_csv(csv_file, smiles_field="smiles", target_fields=["HIV_active"],
                      verbose=verbose, **kwargs)

    def split(self):
        lengths = [int(0.8 * len(self.data)), int(0.1 * len(self.data))]
        lengths += [len(self.data) - sum(lengths)]
        train_set, valid_set, test_set = torch.utils.data.random_split(dataset, lengths, generator=torch.Generator().manual_seed(1141))
        return train_set, valid_set, test_set





dataset = HIV_mols('data/HIV.csv', atom_feature="pretrain", bond_feature="pretrain")

train_set, valid_set, test_set = dataset.split()

model = models.GIN(input_dim=dataset.node_feature_dim,
                   hidden_dims=[1024, 1024, 512, 512],
                   short_cut=True, batch_norm=True, concat_hidden=True)
task = tasks.PropertyPrediction(model, task=("HIV_active",),
                                criterion="bce", metric=("auprc", "auroc", "acc"))
checkpoint = torch.load("clintox_gin_attributemasking_"+str(i)+".pth")["model"]
task.load_state_dict(checkpoint, strict=False)

optimizer = torch.optim.Adam(task.parameters(), lr=1e-3)
solver = core.Engine(task, train_set, valid_set, test_set, optimizer, logger='logging',
                     gpus=[0],
                     batch_size=512)
solver.train(num_epoch=100)
solver.evaluate("valid")
solver.evaluate("test")



#   solver.save("HIV_"+str(i)+".pth")

with open("HIV_gin_pretrain.json", "w") as fout:
    json.dump(solver.config_dict(), fout)
solver.save("HIV_gin_pretrain.pth")

    

print(0)