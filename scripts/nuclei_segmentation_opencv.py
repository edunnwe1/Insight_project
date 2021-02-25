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
    """
        Class to segment nuclei from cell images
        
        Instance variables
        ------------------
        area_threshold : float
            minimum area for nucleus to pass
            
        solidity_threshold : float
            minimum solidity for nucleus to pass
            
        major_minor : float
            maximum major:minor axis ratio for nucleus to pass
            
        ksize : int
            kernel size for contrast-limited adaptive-histogram equalization (CLAHE)
            
        clipLimit : float
            clip limit for CLAHE
            
        croppedFlag : boolean
            whether or not the image is cropped
            
        numClasses : int
            number of classes for multi-otsu thresholding
            
        tile : int
            tile size for local CLAHE
        
        
        Methods
        ------------------
        read_img(image_path)
            reads a color image from image_path
        
        prep_img(color_img)
            performs preprocessing steps on the color image 
        
        get_mask()
            identify the nuclei in the image
            
        get_props()
            get the image properties of the identified nuclei
            
        debris_filter()
            filter nuclei that do not meet requirements
            
        segment_nuclei()
            executes all preceding functions as a full pipeline 
        
        
    """
    
    def __init__(self,area_threshold=250,solidity_threshold=0.6,major_minor=2.5,ksize=3, clipLimit=2.0,cropped_flag=0,numClasses=5,tile=25):
        
        """
            Parameters
            ------------------
            
            area_threshold : float
                minimum area for nucleus to pass

            solidity_threshold : float
                minimum solidity for nucleus to pass

            major_minor : float
                maximum major:minor axis ratio for nucleus to pass

            ksize : int
                kernel size for contrast-limited adaptive-histogram equalization (CLAHE)

            clipLimit : float
                clip limit for CLAHE
                
            croppedFlag : boolean
                whether or not the image is cropped

            numClasses : int
                number of classes for multi-otsu thresholding

            tile : int
                tile size for local CLAHE
                
            color_img : None
                initiates color_img field
                
            image_path : None
                initiates image_path field
                
        """
        
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
        """
            Reads a color image from image_path using cv2.imread
        """
        
        self.color_img = cv2.imread(image_path,1)
        self.image_path = image_path
    
    def prep_img(self,color_img=None):
        """
            Performs pre-processing on the image, including CLAHE and median filtering, using img_preProcessor
        """
        if self.color_img is None:
            self.color_img = color_img
        # convert to gray
        self.original_gray = cv2.cvtColor(self.color_img,cv2.COLOR_BGR2GRAY)
        # process the image
        prep = img_preProcessor(ksize=self.ksize,clipLimit=self.clipLimit,tile=self.tile)
        self.filt_img = prep.apply_CLAHE(prep.apply_med_filt(self.color_img))
        self.gray_img = cv2.cvtColor(self.filt_img,cv2.COLOR_BGR2GRAY)
    
    def get_mask(self):
        """
            Gets the binary mask of detected nuclei, as well as the labeled image
        """
        # compute threshold (to detect nuclei)
#         threshold = filters.threshold_minimum(filt_img) #nuclei are among the darkest parts of the image 
        thresholds = filters.threshold_multiotsu(self.gray_img,classes=self.numClasses)
        threshold = thresholds[0]    
        # binarize
        self.binary_img = self.gray_img < threshold
        # label
        self.labeled_img = measure.label(self.binary_img,background=0)
    
    def get_props(self):
        """
            Extracts the image properties from the labeled image using scikit-image's regionprops
        """
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
        """
            Filters out nuclei that do not meet area, solidity, and major:minor axis ratio criteria 
        """
        # Viable nuclei must meet criteria, else will be rejected as debris
        # threshold by area 
        nuclei_df = nuclei_df[nuclei_df['area']>self.area_threshold]
        # threshold by major:minor axis length
        nuclei_df = nuclei_df[nuclei_df['major_to_minor']<self.major_minor]
        # threshold by solidity
        nuclei_df = nuclei_df[nuclei_df['solidity']>self.solidity_threshold]
        
        return nuclei_df
    
    def segment_nuclei(self,color_img = None):
        """
            Performs the entire segmentation pipeine, returns the data frame of nucleus features
        """
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
        