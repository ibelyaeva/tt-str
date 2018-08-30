import tensorflow as tf
import numpy as np
import t3f
from sklearn.tree.tests.test_tree import y_random
tf.set_random_seed(0)
np.random.seed(0)
import matplotlib.pyplot as plt
import metric_util as mt
import data_util as du
from t3f import shapes
from nilearn import image
from skimage.measure import compare_ssim as ssim
from tensorflow.python.util import nest
import copy
from nilearn import plotting
from t3f import ops
import mri_draw_utils as mrd
from t3f import initializers
from t3f import approximate
from scipy import optimize 
from nilearn.masking import compute_background_mask
from nilearn.masking import compute_epi_mask
from collections import OrderedDict
import pandas as pd
from scipy import stats
from nilearn.image import math_img
import time
import tensor_util as tu
import rimannian_tensor_completion_structural as rtc
import structural_pattern_generator as stp
import ellipsoid_mask as elp

class TensorCompletionStructural(object):
    
    def __init__(self, data_path, observed_ratio, d, n, logger, meta, x0, y0, z0, x_r, y_r, z_r, z_score = 2):
        self.observed_ratio = observed_ratio
        self.missing_ratio = 1.0 - self.observed_ratio
        self.d = d
        self.n = n
        self.logger = logger
        self.meta = meta
        self.z_score = z_score
        
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        
        self.x_r = x_r
        self.y_r = y_r
        self.z_r = z_r 
        self.init_cost_history()
        self.init_dataset(data_path)
        
    def init_dataset(self, path):
        self.x_true_img = mt.read_image_abs_path(path)
        
        self.x_true_data = np.array(self.x_true_img.get_data())
        self.x_true_reshaped = copy.deepcopy(self.x_true_data)
        self.x_true_copy = copy.deepcopy(self.x_true_data)
              
        self.target_shape = mt.get_target_shape(self.x_true_data, self.d)
        self.logger.info("Target Shape: " + str(self.target_shape))
        self.x_true_reshaped_rank = mt.reshape_as_nD(self.x_true_copy, self.d,self.target_shape)
        self.logger.info("D = " + str(self.d) + "; Original Shape: " + str(self.x_true_data.shape) + "; Target Shape: " + str(self.target_shape))
        
        self.tensor_shape = tu.get_tensor_shape(self.x_true_data)
        self.max_tt_rank = tu.get_max_rank(self.x_true_reshaped_rank)
        #self.max_tt_rank = 46
        
        self.logger.info("Tensor Shape: " + str(self.tensor_shape) + "; Max Rank: " + str(self.max_tt_rank))
             
        # mask_indices after reshape
        self.mask_indices = self.init_mask()
        
        #init after reshape
        self.ten_ones = tu.get_ten_ones(self.x_true_reshaped)
       
        # ground truth to be initialized later after reshape
        self.ground_truth, self.norm_ground_truth = tu.normalize_data(self.x_true_reshaped)
        self.logger.info("Norm Ground Truth: " + str(self.norm_ground_truth))
        
        if len(self.x_true_reshaped.shape) > 2:
            self.ground_truth_img = mt.reconstruct_image_affine(self.x_true_img, self.ground_truth)
         
        #initial approximation to be initialized later after reshape
        self.x_init = tu.init_random(self.x_true_reshaped) 
        self.x_init_tcs = self.ground_truth * (1./np.linalg.norm(self.x_init))
        
        self.x_init, self.norm_x_init = tu.normalize_data(self.x_init)
        self.logger.info("Norm X Init: " + str(self.norm_x_init))
                     
        # sparse_observation to be initialized later after reshape
        self.sparse_observation = tu.create_sparse_observation(self.ground_truth, self.mask_indices)
        self.norm_sparse_observation = np.linalg.norm(self.sparse_observation)
        
        self.epsilon = 1e-6
        self.train_epsilon = 1e-5
        self.backtrack_const = 1e-4
        
        # related z_score structures
        self.std_img = None
        self.mean_img = None
        self.z_scored_image = None
        self.ground_truth_z_score = None
        self.mask_z_score_indices = None
        self.mask_z_indices_count = None
        
        
    def init_mask(self):
        self.ellipsoid_mask = self.create_ellipse()
        self.mask_indices = tu.get_mask_with_epi(self.x_true_data, self.x_true_img, self.observed_ratio, self.d)
        
        self.frames_count = int(self.missing_ratio*(self.x_true_data.shape[len(self.x_true_data.shape)-1])/100.0)
        self.frames_count = 10
        self.logger.info("Missing Frames Count: " + str(self.frames_count))
        
        self.x_mask_img = stp.generate_structural_missing_pattern(self.ellipsoid_mask.x0, self.ellipsoid_mask.y0, 
                                                                  self.ellipsoid_mask.z0, self.ellipsoid_mask.x_r, 
                                                                  self.ellipsoid_mask.y_r, self.ellipsoid_mask.z_r,
                                                                             self.frames_count, self.ellipsoid_mask.mask_path)
        
        coords = [self.ellipsoid_mask.x0, self.ellipsoid_mask.y0, self.ellipsoid_mask.z0]
        
        self.x_mask_img_data = np.array(self.x_mask_img.get_data())
        mask_zero_indices_count = np.count_nonzero(self.x_mask_img_data==0)
        self.logger.info("Zero Count 1: " + str(mask_zero_indices_count))
        
        self.mask_indices = np.ones_like(self.x_true_data)
        self.mask_indices[self.x_mask_img_data == 0] = 0.0
        
        mask_zero_indices_count2 = np.count_nonzero(self.mask_indices==0)
        self.logger.info("Zero Count 2: " + str(mask_zero_indices_count2))
                                                            
        mrd.draw_original_vs_reconstructed_rim_z_score(image.index_img(self.x_true_img, 0), image.index_img(self.x_mask_img,0), 
                    image.index_img(self.x_mask_img, 0), "TEST 4D STR",
                    0, self.observed_ratio, 0, 0, 2, coord=coords, folder=self.meta.images_folder, iteration = -1, time=0)
        
                                                                             
        return self.mask_indices
    
    def create_ellipse(self):
        self.ellipsoid_mask  = elp.EllipsoidMask(self.x0, self.y0, self.z0, self.x_r, self.y_r, self.z_r, self.meta.ellipsoid_folder)
        return self.ellipsoid_mask
    
        
    def init_cost_history(self):
        self.rse_cost_history = []
        self.train_cost_history = []
        self.tcs_cost_history = []
        self.tcs_z_scored_history = []
        self.summary_history = []
        
    def complete(self):
        
        self.logger.info("Starting Tensor Completion. Tensor Dimension:" + str(self.d))
        t1 = time.time()
        
        self.original_tensor_shape = tu.get_tensor_shape(self.x_true_data)
        self.original_max_rank = tu.get_max_rank(self.x_true_data)
        self.logger.info("Original Tensor Shape: " + str(self.original_tensor_shape) + "; Original Tensor Max Rank: " + str(self.original_max_rank))
        
        
        if self.d == 2:
            self.complete2D()
        elif self.d == 3:
            self.complete3D()    
        elif self.d == 4:
            self.complete4D()
            #pass
        else:
            errorMsg = "Unknown Tensor Dimensionality. Cannot Complete Image"
            raise(errorMsg)  
        
        t2 = time.time()
        total_time = str(t2 - t1)
        self.logger.info("Finished Tensor Completion. Tensor Dimension:" + str(self.d)  + "... Done.")  
        self.logger.info("Execution time, seconds: " + str(total_time))
        
                
    def complete2D(self):
        self.z_scored_mask = tu.get_z_scored_mask(self.ground_truth_img, 2)
        self.logger.info("Z-score Mask Indices Count: " + str(tu.get_mask_z_indices_count(self.z_scored_mask)))
     
        
        self.rtc_runner = rtc.RiemannianTensorCompletionStructural(self.ground_truth_img,
                                                    self.ground_truth, self.tensor_shape, 
                                                    self.x_init,
                                                    self.mask_indices, 
                                                    self.z_scored_mask,
                                                    self.sparse_observation,
                                                    self.norm_sparse_observation, 
                                                    self.x_init_tcs,
                                                    self.ten_ones, 
                                                    self.max_tt_rank, 
                                                    self.observed_ratio,
                                                    self.epsilon, self.train_epsilon,
                                                    self.backtrack_const, self.logger, 
                                                    self.meta, self.d, self.ellipsoid_mask, self.z_score,
                                                    )
        self.rtc_runner.complete()
        
        mrd.draw_original_vs_reconstructed_rim_z_score(image.index_img(self.ground_truth_img, 0), image.index_img(self.rtc_runner.x_hat_img,0), image.index_img(self.rtc_runner.x_miss_img, 20), "2D fMRI Tensor Completion",
                                             self.rtc_runner.tsc_score, self.observed_ratio, self.rtc_runner.tsc_score, self.rtc_runner.tcs_z_score, 2, coord=None, folder=self.rtc_runner.meta.images_folder,  iteration = -1)
    
        pass
    
    def complete3D(self):
        self.z_scored_mask = tu.get_z_scored_mask(self.ground_truth_img, 2)
        self.logger.info("Z-score Mask Indices Count: " + str(tu.get_mask_z_indices_count(self.z_scored_mask)))

        self.rtc_runner = rtc.RiemannianTensorCompletionStructural(self.ground_truth_img,
                                                    self.ground_truth, self.tensor_shape, 
                                                    self.x_init,
                                                    self.mask_indices, 
                                                    self.z_scored_mask,
                                                    self.sparse_observation,
                                                    self.norm_sparse_observation, 
                                                    self.x_init_tcs,
                                                    self.ten_ones, 
                                                    self.max_tt_rank, 
                                                    self.observed_ratio,
                                                    self.epsilon, self.train_epsilon,
                                                    self.backtrack_const, self.logger, self.meta, self.d, 
                                                    self.ellipsoid_mask,
                                                    self.z_score,
                                                    )
        self.rtc_runner.complete()
        
        mrd.draw_original_vs_reconstructed_rim_z_score(image.index_img(self.ground_truth_img, 0), image.index_img(self.rtc_runner.x_hat_img,0), image.index_img(self.rtc_runner.x_miss_img, 0), "3D fMRI Tensor Completion",
                                             self.rtc_runner.tsc_score, self.observed_ratio, self.rtc_runner.tsc_score, self.rtc_runner.tcs_z_score, 2, coord=None, folder=self.rtc_runner.meta.images_folder,  iteration = -1)
    
    
    def complete4D(self):
        
        self.z_scored_mask = tu.get_z_scored_mask(self.ground_truth_img, 2)
        self.logger.info("Z-score Mask Indices Count: " + str(tu.get_mask_z_indices_count(self.z_scored_mask)))

        self.rtc_runner = rtc.RiemannianTensorCompletionStructural(self.ground_truth_img,
                                                    self.ground_truth, self.tensor_shape, 
                                                    self.x_init,
                                                    self.mask_indices, 
                                                    self.z_scored_mask,
                                                    self.sparse_observation,
                                                    self.norm_sparse_observation, 
                                                    self.x_init_tcs,
                                                    self.ten_ones, 
                                                    self.max_tt_rank, 
                                                    self.observed_ratio,
                                                    self.epsilon, self.train_epsilon,
                                                    self.backtrack_const, self.logger, self.meta, self.d, 
                                                    self.ellipsoid_mask,
                                                    self.z_score,
                                                    )
        self.rtc_runner.complete()
        
        mrd.draw_original_vs_reconstructed_rim_z_score(image.index_img(self.ground_truth_img, 0), image.index_img(self.rtc_runner.x_hat_img,0), image.index_img(self.rtc_runner.x_miss_img, 0), "4D fMRI Tensor Completion",
                                             self.rtc_runner.tsc_score, self.observed_ratio, self.rtc_runner.tsc_score, self.rtc_runner.tcs_z_score, 2, coord=None, folder=self.rtc_runner.meta.images_folder,  iteration = -1)
    
        
        