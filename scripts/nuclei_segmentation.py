import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics as stats

from skimage import io, filters, color,exposure,feature,measure,segmentation
from scipy import ndimage
import cv2 
import imutils
from scipy.spatial import distance as dist

class nuclei_segmenter:
    def __init__(self,area_threshold,p2a_threshold,solidity_threshold,median_filter_param,cropped_flag):
        self.area_threshold = area_threshold
        self.P2A_threshold = p2a_threshold
        self.solidity_threshold = solidity_threshold
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
        nuclei_df['centroid'] = nuclei_df[['centroid-0', 'centroid-1']].values.tolist()
        # threshold by area 
        nuclei_df = nuclei_df[nuclei_df['area']>self.area_threshold]
        # threshold by perimeter-to-area (P2A)
        nuclei_df = nuclei_df[nuclei_df['perimeter']/nuclei_df['area']<self.P2A_threshold]
        # threshold by solidity
        nuclei_df = nuclei_df[nuclei_df['solidity']>self.solidity_threshold]
        # add image path ID 
        nuclei_df['ID'] = image_path
        # if cropped image, limit to cell closest to the center of the image
        if self.cropped_flag:
            img_ctr = [np.round(gray_img.shape[0]/2),np.round(gray_img.shape[1]/2)]
            nuclei_df['ctr_score'] = nuclei_df['centroid'].apply(lambda x: dist.euclidean(x,img_ctr))
            if not nuclei_df.empty:
                nuclei_df = nuclei_df.sort_values(by=['ctr_score']).iloc[0]
            
#             bbox0 = [0.15*gray_img.shape[0], 0.85*gray_img.shape[1]]
#             bbox1 = [0.15*gray_img.shape[1], 0.85*gray_img.shape[1]]
            
#             nuclei_df = nuclei_df[(nuclei_df['centroid-0']>bbox0[0])&(nuclei_df['centroid-0']<bbox0[1])]
#             nuclei_df = nuclei_df[(nuclei_df['centroid-1']>bbox1[0])&(nuclei_df['centroid-1']<bbox1[1])]
            
        return nuclei_df
        