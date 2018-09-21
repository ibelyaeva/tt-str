import texfig

import os
import data_util as du
import configparser
from os import path
import metadata as mdt
import complete_tensor_structural as ct
import structural_pattern_generator as stp


config_loc = path.join('config')
config_filename = 'solution.config'
#config_filename = 'solution-ec2.config'
config_file = os.path.join(config_loc, config_filename)
config = configparser.ConfigParser()
config.read(config_file)
config = config

def complete_random_4D():
    subject_scan_path = du.get_full_path_subject1()
    meta = mdt.Metadata('random', 4)
    root_dir = config.get('log', 'scratch.dir4Dstr')
    solution_dir, movies_folder, images_folder, results_folder, reports_folder, scans_folder, scans_folder_final, scans_folder_iteration = meta.init_meta_data(root_dir)
    
    observed_ratio_list = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    observed_ratio_list = [0.95]
    
    x0, y0, z0 = (2,32,22)
    # size 1
    x_r, y_r, z_r = (7,10,8)
    
    for item in observed_ratio_list:
        current_runner = ct.TensorCompletionStructural(subject_scan_path, item, 4, 1, meta.logger, meta, x0, y0, z0, x_r, y_r, z_r)
        current_runner.complete()
      
    #size 2 
    #x_r, y_r, z_r = (9,10,8)
    #for item in observed_ratio_list:
    #    current_runner = ct.TensorCompletionStructural(subject_scan_path, item, 4, 1, meta.logger, meta, x0, y0, z0, x_r, y_r, z_r)
    #    current_runner.complete()

    #size 3
    #x_r, y_r, z_r = (12,10,8)
    #for item in observed_ratio_list:
    #    current_runner = ct.TensorCompletionStructural(subject_scan_path, item, 4, 1, meta.logger, meta, x0, y0, z0, x_r, y_r, z_r)
    #    current_runner.complete()
        
def complete_random_4Dsize1():
    subject_scan_path = du.get_full_path_subject1()
    meta = mdt.Metadata('structural', 4)
    root_dir = config.get('log', 'scratch.dir4Dstr')
    solution_dir, movies_folder, images_folder, results_folder, reports_folder, scans_folder, scans_folder_final, scans_folder_iteration = meta.init_meta_data(root_dir)
    
    #observed_ratio_list = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    observed_ratio_list = [0.95, 0.9,0.85,0.8,0.75,0.7]
    
    #for item in observed_ratio_list:
    #    mr = 1.0 - item
    #    frame_count = int(mr*144)
    #    ts = stp.create_random_frames(144, frame_count)
    #    meta.logger.info("MR = " + str(mr) + "; " + str(ts.values()) + "; Timepoint Count: " + str(frame_count))
    
    
    x0, y0, z0 = (2,32,22)
    # size 1
    x_r, y_r, z_r = (7,10,8)
    
    for item in observed_ratio_list:
        current_runner = ct.TensorCompletionStructural(subject_scan_path, item, 4, 1, meta.logger, meta, x0, y0, z0, x_r, y_r, z_r)
        current_runner.complete()
        
def complete_random_4Dsize2():
    subject_scan_path = du.get_full_path_subject1()
    meta = mdt.Metadata('random', 4)
    root_dir = config.get('log', 'scratch.dir4Dstr')
    solution_dir, movies_folder, images_folder, results_folder, reports_folder, scans_folder, scans_folder_final, scans_folder_iteration = meta.init_meta_data(root_dir)
    
    observed_ratio_list = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    #observed_ratio_list = [0.95,0.9]
    
    x0, y0, z0 = (2,32,22)
      
    #size 2 
    x_r, y_r, z_r = (9,10,8)
    for item in observed_ratio_list:
        current_runner = ct.TensorCompletionStructural(subject_scan_path, item, 4, 1, meta.logger, meta, x0, y0, z0, x_r, y_r, z_r)
        current_runner.complete()
        
def complete_random_4Dsize3():
    subject_scan_path = du.get_full_path_subject1()
    meta = mdt.Metadata('random', 4)
    root_dir = config.get('log', 'scratch.dir4Dstr')
    solution_dir, movies_folder, images_folder, results_folder, reports_folder, scans_folder, scans_folder_final, scans_folder_iteration = meta.init_meta_data(root_dir)
    
    observed_ratio_list = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    
    x0, y0, z0 = (2,32,22)
  
    #size 3
    x_r, y_r, z_r = (12,10,8)
    for item in observed_ratio_list:
        current_runner = ct.TensorCompletionStructural(subject_scan_path, item, 4, 1, meta.logger, meta, x0, y0, z0, x_r, y_r, z_r)
        current_runner.complete()

def complete_random_3D():
    subject_scan_path = du.get_full_path_subject1()
    meta = mdt.Metadata('random', 3)
    root_dir = config.get('log', 'scratch.dir3D')
    solution_dir, movies_folder, images_folder, results_folder, reports_folder, scans_folder, scans_folder_final, scans_folder_iteration = meta.init_meta_data(root_dir)
    
    observed_ratio_list = [0.9, 0.8, 0.7, 0.6, 0.65, 0.5, 0.4, 0.35, 0.3, 0.2, 0.1]
    observed_ratio_list = [0.9]
    
    x0, y0, z0 = (2, 32,22)
    # size 1
    x_r, y_r, z_r = (7,10,8)
    
    for item in observed_ratio_list:
        current_runner = ct.TensorCompletionStructural(subject_scan_path, item, 3, 0, meta.logger, meta, x0, y0, z0, x_r, y_r, z_r)
        current_runner.complete()

        
def complete_random_2D():
    subject_scan_path = du.get_full_path_subject1()
    meta = mdt.Metadata('random', 2)
    root_dir = config.get('log', 'scratch.dir2D')
    solution_dir, movies_folder, images_folder, results_folder, reports_folder, scans_folder, scans_folder_final, scans_folder_iteration = meta.init_meta_data(root_dir)
    
    observed_ratio_list = [0.9, 0.8, 0.7, 0.6, 0.65, 0.5, 0.4, 0.35, 0.3, 0.2, 0.1]
    observed_ratio_list = [0.25]
    
    x0, y0, z0 = (2, 32,22)
    # size 1
    x_r, y_r, z_r = (7,10,8)
    
    for item in observed_ratio_list:
        current_runner = ct.TensorCompletionStructural(subject_scan_path, item, 2, 1, meta.logger, meta, x0, y0, z0, x_r, y_r, z_r)
        current_runner.complete()
        
def complete_random_4D_by_tr_size():
    subject_scan_path = du.get_full_path_subject1()
    meta = mdt.Metadata('structural', 4)
    root_dir = config.get('log', 'scratch.dir4Dstr')
    solution_dir, movies_folder, images_folder, results_folder, reports_folder, scans_folder, scans_folder_final, scans_folder_iteration = meta.init_meta_data(root_dir)
    
    #observed_ratio_list = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    observed_ratio_list = [0.95, 0.9,0.85,0.8,0.75,0.7]
    
    x0, y0, z0 = (2,32,22)
    # size 1
    
    volumes_list = []
    volumes_list.append((8,10,8))
    volumes_list.append((9,10,8))
    volumes_list.append((10,10,8))
    volumes_list.append((11,10,8))
    volumes_list.append((12,10,8))
    
    #x_r, y_r, z_r = (7,10,8)
    
    for item in observed_ratio_list:
        for el in volumes_list:
            solution_dir, movies_folder, images_folder, results_folder, reports_folder, scans_folder, scans_folder_final, scans_folder_iteration = meta.init_meta_data(root_dir)
            current_runner = ct.TensorCompletionStructural(subject_scan_path, item, 4, 1, meta.logger, meta, x0, y0, z0, el)
            current_runner.complete()
        



if __name__ == "__main__":
    pass
    #complete_random_2D()
    #complete_random_4Dsize1()
    complete_random_4D_by_tr_size()
    #complete_random_4Dsize2()
    #complete_random_3D()

    
    
