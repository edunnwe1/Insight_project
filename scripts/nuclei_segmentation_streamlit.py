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
    def __init__(self,area_threshold=325,p2a_threshold=0.25,solidity_threshold=0.9,ksize=9, clipLimit=2.0,cropped_flag=0):
        self.area_threshold = area_threshold
        self.P2A_threshold = p2a_threshold
        self.solidity_threshold = solidity_threshold
        self.ksize = ksize
        self.clipLimit=clipLimit
        self.cropped_flag = cropped_flag
        
    def segment_nuclei(self,color_img):
        ###### pre-processing #####
        gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        # apply median filter
        img = cv2.medianBlur(gray_img,self.ksize)
        # contrast-limited adaptive histogram equalization
        tilesize = [np.round(gray_img.shape[0]/25),np.round(gray_img.shape[1]/25)]
        clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=(tilesize[0].astype('int'),tilesize[0].astype('int')))
        cl1 = clahe.apply(img)
        filt_img = cl1
        
        ##### segmentation #####
        # compute threshold
        threshold = filters.threshold_minimum(filt_img) #nuclei are among the darkest parts of the image 
        # binarize
        binary_img = filt_img < threshold
        # label
        labeled_img = measure.label(binary_img,background=0)
        # extract image properties
        props_table = measure.regionprops_table(labeled_img,intensity_image=gray_img, properties=['area','centroid','extent','perimeter','eccentricity','solidity','mean_intensity','bbox'])
        nuclei_df = pd.DataFrame.from_dict(props_table)
        nuclei_df['centroid'] = nuclei_df[['centroid-0', 'centroid-1']].values.tolist()
        # threshold by area 
        nuclei_df = nuclei_df[nuclei_df['area']>self.area_threshold]
        # threshold by perimeter-to-area (P2A)
        nuclei_df = nuclei_df[nuclei_df['perimeter']/nuclei_df['area']<self.P2A_threshold]
        # threshold by solidity
        nuclei_df = nuclei_df[nuclei_df['solidity']>self.solidity_threshold]
        # add intensity by channel 
        nuclei_df['meanI_R'] = [np.mean(color_img[labeled_img==i,0]) for i in range(len(nuclei_df))]
        nuclei_df['meanI_B'] = [np.mean(color_img[labeled_img==i,1]) for i in range(len(nuclei_df))]
        nuclei_df['meanI_G'] = [np.mean(color_img[labeled_img==i,2]) for i in range(len(nuclei_df))]
        
        # if cropped image, limit to cell closest to the center of the image
        if self.cropped_flag:
            img_ctr = [np.round(filt_img.shape[0]/2),np.round(filt_img.shape[1]/2)]
            nuclei_df['ctr_score'] = nuclei_df['centroid'].apply(lambda x: dist.euclidean(x,img_ctr))
            if not nuclei_df.empty:
                nuclei_df = nuclei_df.sort_values(by=['ctr_score']).iloc[0]
            
        return nuclei_df,filt_img
        