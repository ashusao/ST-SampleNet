# ST-SampleNet

# Spatially Constrained Transformer with Efficient Global Relation Modelling for Spatio-Temporal Prediction

## Description
This repository is the implementation of the paper "**Spatially Constrained Transformer with Efficient Global Relation Modelling for Spatio-Temporal Prediction**" by Ashutosh Sao and Simon Gottschalk. 

![ST-SampleNet](images/Architecture.png)

## Installation
To install all dependencies
```bash
conda create -n stsamplenet
conda activate stsamplenet
bash install.sh
```

# Folder Structure 
Input data: `tmp/data/`

Saved model: `tmp/model/`

# Training & Evaluation
Run teacher model containing all regions first.

Run:
```bash
python3 main.py
```
To train region pruned model set the `region_keep_rate` and `train_teacher` flag in `config.ini` 
and train the model again using the command above to see the effect of pruning.
