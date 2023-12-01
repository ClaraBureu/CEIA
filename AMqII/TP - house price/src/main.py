"""
main.py

This script serves as a command-line interface to execute different modes of pipelines based on user input.

Imports:
- argparse for parsing command-line arguments
- subprocess for running external Python scripts

Usage:
- Execute this script with a specified mode ('train' or 'predict') to run the corresponding pipeline.
- The script uses 'argparse' to parse the command-line argument specifying the execution mode.

Functionality:
- The script parses the command-line argument to determine the mode of execution ('train' or 'predict').
- If the mode is 'train', it triggers the 'train_pipeline.py' subprocess to execute the training pipeline.
- If the mode is 'predict' (or any other mode), it triggers the 'inference_pipeline.py' subprocess to execute the inference pipeline.

Note:
- Ensure the existence of 'train_pipeline.py' and 'inference_pipeline.py' in the 'src' directory.
- Use this script as a control mechanism to trigger different pipelines based on specified modes.
- The 'check=False' argument in subprocess.run() suppresses errors; ensure script correctness and handle exceptions accordingly.

DESCRIPCIÓN: main.py
AUTOR: Clara Bureu - Maximiliano Medina - Luis Pablo Segovia
FECHA: 01/12/2023
"""

# Imports
import argparse
import logging
import subprocess

# Log setting
logging.basicConfig(level=logging.DEBUG, 
                    filename='data_logger.log', 
                    filemode='a', 
                    format='%(asctime)s:%(levelname)s:%(message)s')


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, help='Execution mode: train or predict')
    args = parser.parse_args()

    # Retrieve the specified mode from command-line arguments
    mode = args.mode

    # Starting log
    logging.info('main.py starting.')

    # Define the mode of usage of the file
    if mode == 'train':
        subprocess.run(['Python', 'src/train_pipeline.py'], check=False)
    else:
        subprocess.run(['Python', 'src/inference_pipeline.py'], check=False)

    # End log
    logging.info('main.py end.')