import numpy as np
import sys
import os
import torch


def get_sorted_eeg_and_label_files(data_dir):
    """
    Function to list and sort EEG and label files in the given directory.
    
    Args:
    - data_dir (str): Path to the directory containing EEG and label files.
    
    Returns:
    - tuple: Sorted lists of EEG files and label files
    """
    # List all files in the directory
    all_files = os.listdir(data_dir)
    
    # Filter and sort EEG and label files
    eeg_files = sorted([f for f in all_files if f.startswith("eeg") and f.endswith(".npy")], key=lambda x: int(x[3:-4]))
    label_files = sorted([f for f in all_files if f.startswith("label") and f.endswith(".npy")], key=lambda x: int(x[5:-4]))
    
    return eeg_files, label_files
    
    
def z_score_normalize(data):
    mean = np.mean(data)  
    std = np.std(data)    
    normalized_data = (data - mean) / (std + 1e-8)  
    #print(normalized_data.shape)
    return normalized_data
    
    
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model




