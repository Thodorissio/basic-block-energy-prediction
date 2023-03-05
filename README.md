# Overview

This repo was developed in the scope of basic block energy consumption prediction thesis for the school of Electrical and Computer Engineering of National Technical University of Athens. It aims to predict the energy that a basic block of code will consume based on custom created dataset.

# Usage

In order to use the energy prediction CLI tool the next steps should be followed:


* Create a conda environment with python 3.10.4:

```
conda create -n energy-prediction python=3.10.4 ipython
```

* Install package using setup.py:

```
cd code-energy-prediction
conda activate energy-prediction
pip install .
```

* Create a txt containg the basic blocks, for which the energy consumption measurements are desired. The txt should use the example formate:
```
@bb_0
cmpw (%r12) %bp
jnz
@bb_1
movl %eax
orl %eax
orl %eax
jz 0x932
```

* Measure the energy consumption using the cli tool:
```
cd bb_energy_prediction
conda activate energy-prediction
python energy_prediction.py --bbs_path "/path/to/bb.txt" --results_save_dir "/path/to/save_results"
```

The results energy units are 61μJ following the Skylake technology.

# Developer Usage

For developer usage, the following requirements are presented aside from the ones alredy mentioned:

* Install jupyter kernel:
```
conda activate energy-prediction
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=energy-prediction
```

* Download the [PalmTree library](https://github.com/palmtreemodel/PalmTree) fork in order to use the PalmTree Transformer:
```
git clone https://github.com/Thodorissio/pre-trained-palmtree.git
cd ./pre-trained-palmtree
conda activate energy-prediction
pip install .
```

* The custom dataset the models are based on is available [here](https://github.com/jimbou/energy_dataset). It should be cloned order to train models and make predictions:
```
git clone https://github.com/jimbou/energy_dataset.git
```

* Create a .env file following the .env.example

* Read the [demo notebook](https://github.com/)

# Documentation
documentation is available [here](https://thodorissio.github.io/code-energy-prediction/)
# TODO
Create Documentation
Add thesis document
Add results png

# Contributors

* Siozos Theodoros
