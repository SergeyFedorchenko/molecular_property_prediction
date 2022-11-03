# Molecular Property Prediction

### Random Forest
Molecular features were constructed with RDKit descriptors and fingerprints. Then cross-validated hyperparameter searches were applied.




### Graph Convolutional Model
GNNs model based on https://arxiv.org/pdf/1810.00826.pdf 
Hidden architectures were chosen by cross-validated hyperparameter search. 
Pretraining by learning the regularities of the node/edge attributes distributed over the graph structure increased performance.

### Results

See exploration.ipynb or https://nbviewer.org/github/SergeyFedorchenko/molecular_property_prediction/blob/main/exploration.ipynb
