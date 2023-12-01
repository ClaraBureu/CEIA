"""
train_pipeline.py

This script automates the feature engineering and training process using subprocess.

Usage:
- Execute this script to run feature engineering on the 'train' data and subsequently run the training process.
- Ensure the necessary Python scripts ('feature_engineering.py' and 'train.py') exist in the 'src' directory.

Description:
- This script uses the 'subprocess' module to execute separate Python scripts responsible for feature engineering and training.
- The first subprocess call runs the 'feature_engineering.py' script, processing the 'train' data.
- The second subprocess call executes the 'train.py' script, which likely contains the model training logic.

Note:
- Ensure Python is installed and accessible in the environment.
- Adjust paths or script names in the subprocess calls if located in different directories or named differently.

Caution:
- The 'check=False' argument in subprocess.run() suppresses errors. Ensure script correctness and handle exceptions accordingly.

DESCRIPCIÓN: train_pipeline.py
AUTOR: Clara Bureu - Maximiliano Medina - Luis Pablo Segovia
FECHA: 01/12/2023
"""

import subprocess

subprocess.run(['Python', 'src/feature_engineering.py', 'train'], check=False)

subprocess.run(['Python', 'src/train.py'], check=False)
