import numpy as np
import csv
import pandas as pd

def read_tsc_cost(file_path):
    col_names = ['k', 'mr', 'ts_count', 'el_volume', 'roi_volume', 'tsc_cost']
    dtype_names = {'k': np.int32, 'mr': np.float32, 'ts_count': np.int32, 'el_volume': np.float32, 'roi_volume': np.int32, 'tsc_cost': np.float64 }
    result = pd.read_csv(file_path, sep=',', usecols=col_names, dtype=dtype_names)
    return result

def read_tsc_z_cost(file_path):
    col_names = ['k', 'mr', 'ts_count', 'el_volume', 'roi_volume', 'tsc_z_cost']
    dtype_names = {'k': np.int32, 'mr': np.float32, 'ts_count': np.int32, 'el_volume': np.float32, 'roi_volume': np.int32, 'tsc_z_cost': np.float64 }
    result = pd.read_csv(file_path, sep=',', usecols=col_names, dtype=dtype_names)
    return result

def read_rse_cost(file_path):
    col_names = ['k', 'mr', 'ts_count', 'el_volume', 'roi_volume', 'rse_cost']
    dtype_names = {'k': np.int32, 'mr': np.float32, 'ts_count': np.int32, 'el_volume': np.float32, 'roi_volume': np.int32, 'rse_cost': np.float64 }
    result = pd.read_csv(file_path, sep=',', usecols=col_names, dtype=dtype_names)
    return result

def read_train_cost(file_path):
    col_names = ['k', 'mr', 'ts_count', 'el_volume', 'roi_volume', 'train_cost']
    dtype_names = {'k': np.int32, 'mr': np.float32, 'ts_count': np.int32, 'el_volume': np.float32, 'roi_volume': np.int32, 'train_cost': np.float64 }
    result = pd.read_csv(file_path, sep=',', usecols=col_names, dtype=dtype_names)
    return result

def read_solution_cost(file_path):
    col_names = ['k', 'mr', 'ts_count', 'el_volume', 'roi_volume', 'cost']
    dtype_names = {'k': np.int32, 'mr': np.float32, 'ts_count': np.int32, 'el_volume': np.float32, 'roi_volume': np.int32, 'cost': np.float64 }
    result = pd.read_csv(file_path, sep=',', usecols=col_names, dtype=dtype_names)
    return result

def read_data_structural(file_path):
    result = pd.read_csv(file_path, sep=',')
    return result

def read_data_by_path(file_path):
    result = pd.read_csv(file_path, sep=',')
    return result
