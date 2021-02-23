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
from image_preprocessing import img_preProcessor

class nuclei_segmenter:
    def __init__(self,area_threshold=250,solidity_threshold=0.6,major_minor=2.5,ksize=3, clipLimit=2.0,cropped_flag=0,numClasses=5,tile=25):
        self.area_threshold = area_threshold
        self.major_minor = major_minor
        self.solidity_threshold = solidity_threshold
        self.ksize = ksize
        self.clipLimit=clipLimit
        self.cropped_flag = cropped_flag
        self.numClasses = numClasses
        self.tile = tile
        self.color_img = None
        self.image_path = None
    
    def read_img(self,image_path):
        self.color_img = cv2.imread(image_path,1)
        self.image_path = image_path
    
    def prep_img(self,color_img=None):
        if self.color_img is None:
            self.color_img = color_img
        # convert to gray
        self.original_gray = cv2.cvtColor(self.color_img,cv2.COLOR_BGR2GRAY)
        # process the image
        prep = img_preProcessor(ksize=self.ksize,clipLimit=self.clipLimit,tile=self.tile)
        self.filt_img = prep.apply_CLAHE(prep.apply_med_filt(self.color_img))
        self.gray_img = cv2.cvtColor(self.filt_img,cv2.COLOR_BGR2GRAY)
    
    def get_mask(self):
        # compute threshold (to detect nuclei)
#         threshold = filters.threshold_minimum(filt_img) #nuclei are among the darkest parts of the image 
        thresholds = filters.threshold_multiotsu(self.gray_img,classes=self.numClasses)
        threshold = thresholds[0]    
        # binarize
        self.binary_img = self.gray_img < threshold
        # label
        self.labeled_img = measure.label(self.binary_img,background=0)
    
    def get_props(self):
        # extract image properties
        props_table = measure.regionprops_table(self.labeled_img,intensity_image=self.original_gray, properties=['area','centroid','major_axis_length','minor_axis_length','perimeter','eccentricity','solidity','mean_intensity','bbox'])
        nuclei_df = pd.DataFrame.from_dict(props_table)
        nuclei_df['centroid'] = nuclei_df[['centroid-0', 'centroid-1']].values.tolist()
        # add intensity by channel - BGR because openCV
        nuclei_df['meanI_B'] = [np.mean(self.color_img[self.labeled_img==i,0]) for i in range(len(nuclei_df))]
        nuclei_df['meanI_G'] = [np.mean(self.color_img[self.labeled_img==i,1]) for i in range(len(nuclei_df))]
        nuclei_df['meanI_R'] = [np.mean(self.color_img[self.labeled_img==i,2]) for i in range(len(nuclei_df))]
        # add major:minor axis ratio
        nuclei_df['major_to_minor'] = nuclei_df['major_axis_length']/nuclei_df['minor_axis_length']
        
        return nuclei_df
    
    def debris_filter(self,nuclei_df):
        # Viable nuclei must meet criteria, else will be rejected as debris
        # threshold by area 
        nuclei_df = nuclei_df[nuclei_df['area']>self.area_threshold]
        # threshold by major:minor axis length
        nuclei_df = nuclei_df[nuclei_df['major_to_minor']<self.major_minor]
        # threshold by solidity
        nuclei_df = nuclei_df[nuclei_df['solidity']>self.solidity_threshold]
        
        return nuclei_df
    
    def segment_nuclei(self,color_img = None):
        # prep image
        self.prep_img(color_img)
        
        # get labeled nuclei
        self.get_mask()
        
        # extract nucleus features
        nuclei_df = self.get_props()
        
        # remove debris
        nuclei_df = self.debris_filter(nuclei_df)
        
        # add image path ID 
        nuclei_df['ID'] = self.image_path
        
        # if cropped image, limit to cell closest to the center of the image
        if self.cropped_flag:
            img_ctr = [np.round(self.filt_img.shape[0]/2),np.round(self.filt_img.shape[1]/2)]
            nuclei_df['ctr_score'] = nuclei_df['centroid'].apply(lambda x: dist.euclidean(x,img_ctr))
            if not nuclei_df.empty:
                nuclei_df = nuclei_df.sort_values(by=['ctr_score']).iloc[0]    
#         if nuclei_df.empty:
#             nuclei_df = pd.DataFrame(nuclei_df.apply(lambda x: np.nan)).transpose()

        return nuclei_df
        