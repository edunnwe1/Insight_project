import numpy as np
# from skimage import io, filters, color,exposure,feature,measure,segmentation
# from skimage.restoration import denoise_nl_means, estimate_sigma
# from skimage import img_as_ubyte, img_as_float
# from scipy import ndimage
import cv2 

class img_preProcessor:
    
    """
        This class performs image processing on a color image (median filtering, CLAHE)
        
        Instance Variables
        ------------------
        ksize : int
            size of the kernel for contrast-limited adaptive histogram equalization (CLAHE)
            
        clipLimit : int
            clip limit for CLAHE
            
        tile : int
            size of the tile for CLAHE
        
        Methods
        ------------------
        read_img(image_path)
            reads the color image from image_path using cv2.imread
            
        apply_CLAHE(color_img)
            applies CLAHE to the image
            
        apply_med_filt(color_img)
            applies median filtering to the image
        
    """
    
    def __init__(self,ksize=5,clipLimit=2.0,tile=25):
        
        """
        Parameters
        ------------------
        ksize : int
            size of the kernel for contrast-limited adaptive histogram equalization (CLAHE)
            
        clipLimit : int
            clip limit for CLAHE
            
        tile : int
            size of the tile for CLAHE
            
        CLAHE_flag : boolean
            whether or not CLAHE was performed
            
        med_flag : boolean
            whether or not median filtering was performed
        """

        self.ksize = ksize # kernel size for the median filter
        self.clipLimit=clipLimit # clip limit for the CLAHE
        self.tile = 25
        self.CLAHE_flag = 0 
        self.med_flag = 0
        
    def read_img(self,image_path):
        color_img = cv2.imread(image_path,1) # need to make sure we read with opencv to use other opencv functions. note: opencv is BGR
        return color_img
    
    def apply_CLAHE(self,color_img):
        """
            Applies CLAHE to color_img
        """
        # convert the image into LAB - first channel (L) becomes a luminance only channel
        lab_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2LAB)
        # split the channels
        l,a,b = cv2.split(lab_img)
        # apple CLAHE to L channel
        tilesize = [np.round(color_img.shape[0]/self.tile),np.round(color_img.shape[1]/self.tile)] # tile size for local histograms, done this way to be consistent across cluster vs not cluster images 
        clahe = cv2.createCLAHE(clipLimit=self.clipLimit,tileGridSize=(tilesize[0].astype('int'),tilesize[0].astype('int')))
        clahe_img = clahe.apply(l)
        # combine the CLAHE enhanced L-channel back with the A and B channels
        lab_img2 = cv2.merge((clahe_img,a,b))
        # convert back to color
        CLAHE_img = cv2.cvtColor(lab_img2,cv2.COLOR_LAB2BGR)
        self.CLAHE_flag = 1
        return CLAHE_img
        
    def apply_med_filt(self,color_img):
        """
            Applies median filtering to color_img
        """
        # convert the image to grayscale
        gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        # apply median filter
        med_img = cv2.medianBlur(gray_img,self.ksize)
        self.med_flag = 1
        return cv2.cvtColor(med_img, cv2.COLOR_GRAY2BGR)
        
        
    
    