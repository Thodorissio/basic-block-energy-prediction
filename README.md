# Overview

This is my thesis repo. It aims to predict the energy that a basic block of code will consume based on custom created dataset.

# Install

You will be required to download the [PalmTree library](https://github.com/palmtreemodel/PalmTree) fork in order to use the PalmTree Transformer:
```
git clone https://github.com/Thodorissio/pre-trained-palmtree.git
cd ./pre-trained-palmtree
pip install .
```

The custom dataset the models are based on is available [here](https://github.com/jimbou/energy_dataset). It should be cloned order to train models and make predictions:
```
git clone https://github.com/jimbou/energy_dataset.git
```

Install package setup.py:

```
cd code-energy-prediction
pip install .
```

# To do

* Run final optuna studies for models and regressors
* Clean Regressors code
* Create predict function that takes bb as argument
* Make repo into module

# Contributors

* Siozos Theodoros
