import subprocess

subprocess.run(['Python', 'feature_engineering.py', 'train'], check=False)

subprocess.run(['Python', 'train.py'], check=False)