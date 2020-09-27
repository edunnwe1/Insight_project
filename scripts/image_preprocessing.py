import numpy as np
from skimage import io, filters, color,exposure,feature,measure,segmentation
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_ubyte, img_as_float
from scipy import ndimage
import cv2 

class img_preProcessor:
    
    def __init__(self,ksize=5,clipLimit=2.0,radius=30):
        self.ksize = ksize # kernel size for the median filter
        self.clipLimit=clipLimit # clip limit for the CLAHE
        self.radius = radius # radius for rolling ball
        self.CLAHE_flag = 0 
        self.med_flag = 0
        self.NLM_flag = 0
        self.rolling_flag = 0
        
    def read_img(self,image_path):
        color_img = cv2.imread(image_path,1) # need to make sure we read with opencv to use other opencv functions. note: opencv is BGR
        return color_img
    
    def apply_CLAHE(self,color_img):
        # convert the image into LAB - first channel (L) becomes a luminance only channel
        lab_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2LAB)
        # split the channels
        l,a,b = cv2.split(lab_img)
        # apple CLAHE to L channel
        tilesize = [np.round(color_img.shape[0]/25),np.round(color_img.shape[1]/25)] # tile size for local histograms, done this way to be consistent across cluster vs not cluster images 
        clahe = cv2.createCLAHE(clipLimit=self.clipLimit,tileGridSize=(tilesize[0].astype('int'),tilesize[0].astype('int')))
        clahe_img = clahe.apply(l)
        # combine the CLAHE enhanced L-channel back with the A and B channels
        lab_img2 = cv2.merge((clahe_img,a,b))
        # convert back to color
        CLAHE_img = cv2.cvtColor(lab_img2,cv2.COLOR_LAB2BGR)
        self.CLAHE_flag = 1
        return CLAHE_img
        
    def apply_med_filt(self,color_img):
        # convert the image to grayscale
        gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        # apply median filter
        med_img = cv2.medianBlur(gray_img,self.ksize)
        self.med_flag = 1
        return med_img
        
    def apply_rolling_ball(self,color_img):
        # deprecated: too slow
        pass
    
#         from cv2_rolling_ball import subtract_background_rolling_ball
#         # convert the image to grayscale
#         gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
#         # apply rolling ball 
#         final_img, background = subtract_background_rolling_ball(gray_img, self.radius, light_background=True,
#                                              use_paraboloid=False, do_presmooth=True)
#         return final_img, background
        
    
    