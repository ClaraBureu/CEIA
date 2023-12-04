"""
inference_pipeline.py

This script executes feature engineering and prediction modules using subprocess.

Usage:
    - Ensure Python is installed and accessible in the environment.
    - Make sure the necessary modules 'feature_engineering.py' and 'predict.py' 
    exist in the 'src' directory.

Example:
    - To run feature engineering on 'test' data and then perform prediction:
        python this_script.py

DESCRIPCIÓN: inference_pipeline.py
AUTOR: Clara Bureu - Maximiliano Medina - Luis Pablo Segovia
FECHA: 01/12/2023
"""

import subprocess

subprocess.run(['Python', 'src/feature_engineering.py', 'test'], check=False)

subprocess.run(['Python', 'src/predict.py'], check=False)
