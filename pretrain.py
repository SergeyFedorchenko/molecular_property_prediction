from torchdrug import core, models, tasks
from torch import optim
from graph_conv_utils import HIV_mols
import json

dataset = HIV_mols('data/HIV.csv', atom_feature="pretrain", bond_feature="pretrain")

model = models.GIN(input_dim=dataset.node_feature_dim,
                   hidden_dims=[1024, 1024, 512, 512],
                   edge_input_dim=dataset.edge_feature_dim,
                   batch_norm=True, readout="mean")
task = tasks.AttributeMasking(model, mask_rate=0.15)

optimizer = optim.Adam(task.parameters(), lr=1e-3)
solver = core.Engine(task, dataset, None, None, optimizer, 
                     batch_size=256)

solver.train(num_epoch=100)

with open("HIV_gin_pretrained.json", "w") as fout:
    json.dump(solver.config_dict(), fout)
solver.save("HIV_gin_pretrained.pth")