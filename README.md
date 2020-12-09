# XDebugger4ML

This repository contains the implementation source code of the following paper:

Explainable Debugger for Black-box Machine Learning Models

# Setup
1- Clone the repository using HTTP/SSH:
```
git clone https://github.com/peymanras/XDebugger4ML
```
2- Create a conda virtual environment:
```
conda create -n XDebugger4ML python=3.6
```
3- Activate the conda environment: 
```
conda activate XDebugger4ML
```
4- Standing in XDebugger4ML directory, install the requirements:
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
python neighborhood_influence.py
```
2- To reproduce the occurrence histogram results run:
```
python occurrence_histogram.py
```
3- To explain an instance using quasi-global explanation method run:
```
python quasi_global_explanation.py
```
4- To reproduce the diversity results of quasi-global explanations run:
```
python quasi_global_explanation_diversity.py
```
5- To explain an instance globally run:
```
python global_explanation.py
```
6- To visualize feature values vs. contribution values run:
```
python data_visualization.py
```
