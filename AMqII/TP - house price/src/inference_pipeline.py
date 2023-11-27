import subprocess

subprocess.run(['Python', 'feature_engineering.py', 'test'], check=False)

subprocess.run(['Python', 'predict.py'], check=False)