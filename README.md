# Overview

This is my thesis repo. It aims to predict the energy that a basic block of code will consume based on custom created dataset.

# Basic Dependencies

You will be required to download the [PalmTree library](https://github.com/palmtreemodel/PalmTree) fork in order to use the PalmTree Transformer:
```
git clone https://github.com/Thodorissio/pre-trained-palmtree.git
cd ./pre-trained-palmtree
pip install .
```

environment.yml will be uploaded as soon as the major developement has been completed

```
cd code-energy-prediction
conda env create -f environment.yml
conda activate energy-prediction
```

# To do

* Run final optuna studies for models and regressors
* Clean Regressors code
* Create predict function that takes bb as argument
* Make repo into module

# Contributors

* Siozos Theodoros
