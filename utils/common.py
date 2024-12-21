import os
import inspect
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataloader.dataset import LungNoduleDataset


def save_config_to_yaml(config_obj, parrent_dir: str):
    """
    Saves the given Config object as a YAML file. The output file name is derived
    from the module name where the Config class is defined.

    Args:
        config_obj (Config): The Config object to be saved.
        module_name (str): Name of the Python module containing the Config class definition.
    """
    os.makedirs(parrent_dir, exist_ok=True)
    # Get the source file path for the specified module
    module_path = inspect.getfile(config_obj)
    
    # Extract the base file name without extension
    base_filename = os.path.splitext(os.path.basename(module_path))[0]

    # Construct the output YAML file name
    output_file = f"{base_filename}.yaml"
    output_file = os.path.join(parrent_dir, output_file)

    # Convert the Config object to a dictionary
    config_dict = config_obj.__dict__
    config_dict = {k: v for k, v in config_obj.__dict__.items() if not k.startswith('__')}
    # Save the dictionary as a YAML file
    with open(output_file, 'w') as yaml_file:
        yaml.dump(config_dict, yaml_file, sort_keys=False)

    return config_dict

def get_parameters(fn, original_dict):
    new_dict = dict()
    arg_names = inspect.getfullargspec(fn)[0]
    for k in original_dict.keys():
        if k in arg_names:
            new_dict[k] = original_dict[k]
    return new_dict

def get_dataloader(config):
    csv_data = pd.read_csv(config.csv_path)
    data_dir = config.data_dir
    subject_ids = csv_data['Subject ID'].unique()
    train_ids, val_ids = train_test_split(subject_ids, test_size=config.spilt_size, random_state=42)
    train_data = csv_data[csv_data['Subject ID'].isin(train_ids)]
    val_data = csv_data[csv_data['Subject ID'].isin(val_ids)]
        
    train_dataset = LungNoduleDataset(train_data, data_dir, normalize=True)
    val_dataset = LungNoduleDataset(val_data, data_dir, normalize=True)

    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.eval_batch_size, shuffle=False)

    return train_loader, val_loader
