import os

from torchdrug import data, utils
from torchdrug.core import Registry as R


@R.register("datasets.ClinTox")
@utils.copy_args(data.MoleculeDataset.load_csv, ignore=("smiles_field", "target_fields"))
class ClinTox(data.MoleculeDataset):
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

    url = "http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/clintox.csv.gz"
    md5 = "db4f2df08be8ae92814e9d6a2d015284"
    target_fields = ["FDA_APPROVED", "CT_TOX"]

    def __init__(self, path, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        zip_file = utils.download(self.url, path, md5=self.md5)
        csv_file = utils.extract(zip_file)

        self.load_csv(csv_file, smiles_field="smiles", target_fields=self.target_fields,
                      verbose=verbose, **kwargs)


import torch
from torchdrug import core, models, tasks


dataset = ClinTox('data/clinTox')
train_set, valid_set, test_set = dataset.split()

model = models.GIN(input_dim=dataset.node_feature_dim,
                   hidden_dims=[256, 256, 256, 256],
                   short_cut=True, batch_norm=True, concat_hidden=True)
task = tasks.PropertyPrediction(model, task=("Photosensitation",),
                                criterion="bce", metric=("auprc", "auroc"))

optimizer = torch.optim.Adam(task.parameters(), lr=1e-3)
solver = core.Engine(task, train_set, valid_set, test_set, optimizer,
                     #gpus=[0],
                     batch_size=1024)
solver.train(num_epoch=100)
solver.evaluate("valid")
solver.evaluate("test")