# ExplainableDebugger

This repository contains implementation codes of the following paper:

Explainable Debugger for Black-box Machine Learning Models

# Setup
1- Clone the repository using HTTP/SSH:
```
git clone https://github.com/peymanras/ExplainableDebugger
```
2- Create a conda virtual environment:
```
conda create -n ExplainableDebugger python=3.6
```
3- Activate the conda environment: 
```
conda activate ExplainableDebugger
```
4- Standing in ExplainableDebugger directory, install the requirements:
```
pip install -r requirements.txt
```
5- Install the Accumulated Local Effects (ALE) package:
```
pip install git+https://github.com/MaximeJumelle/ALEPython.git@dev#egg=alepython
```
6- Install SHAP package:
```
pip install git+https://github.com/slundberg/shap.git
```
7- Run initial setup:
```
python setup.py
```
8- Install TBB library required by EXPLAN:
```
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install libtbb2 

# CentOS
sudo yum update
sudo yum install tbb
```

# Reproducing the results
1- To reproduce the neighborhood influence results run:
```
python neighborhood_influence_experiment.py
```
2- To reproduce the occurrence distribution results run:
```
python occurrence_distribution_experiment.py
```
3- To reproduce the local explanation similarity results run:
```
python local_explanation_similarity_experiment.py
```
4- To explain an anomaly instance locally using EXPLAN method run:
```
python local_explanation_experiment.py
```
5- To explain an anomaly instance globally using ALE plots run:
```
python global_explanation_experiment.py
```
6- To visualize feature values vs. contribution values run:
```
python data_visualization.py
```
