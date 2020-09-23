import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics as stats

from skimage import io, filters, color,exposure,feature,measure,segmentation
from scipy import ndimage
import cv2 
import imutils

class nuclei_segmenter:
    def __init__(self,area_threshold,median_filter_param,cropped_flag):
        self.area_threshold = area_threshold
        self.filter_param = median_filter_param
        self.cropped_flag = cropped_flag
        
    def segment_nuclei(self,image_path):
        # read the image
        image = plt.imread(image_path)
        ###### pre-processing #####
        # convert to grayscale
        gray_img = color.rgb2gray(image)
        # apply median filter
        filt_img = ndimage.median_filter(gray_img,self.filter_param)
        ##### segmentation #####
        # compute threshold
        threshold = filters.threshold_minimum(filt_img) #nuclei are among the darkest parts of the image 
        # binarize
        binary_img = filt_img < threshold
        # label
        labeled_img = measure.label(binary_img,background=0)
        # extract image properties
        props_table = measure.regionprops_table(labeled_img,intensity_image=gray_img, properties=['area','centroid','major_axis_length','minor_axis_length','perimeter','eccentricity','solidity','max_intensity','min_intensity','bbox'])
        nuclei_df = pd.DataFrame.from_dict(props_table)
        # threshold by area 
        nuclei_df = nuclei_df[nuclei_df['area']>self.area_threshold]
        nuclei_df['ID'] = image_path
        # limit by bounding box (if cropped_flag - to validate the segmentation)
        if self.cropped_flag:
            self.bbox_lower = 0.1*np.min(gray_img.shape)
            self.bbox_upper = 0.9*np.min(gray_img.shape)
            nuclei_df = nuclei_df[(nuclei_df.filter(regex='^bbox',axis=1).head().gt(self.bbox_lower) & nuclei_df.filter(regex='^bbox',axis=1).head().lt(self.bbox_upper)).sum(axis=1)==4]
        return nuclei_df
        