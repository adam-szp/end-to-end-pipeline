import yaml

def load_config(path='configs/config.yaml'):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_hash(path='data/raw/diabetes.csv.dvc'):
    with open(path, 'r') as f:
        hash_value = yaml.safe_load(f)['outs'][0]['md5']
    return hash_value