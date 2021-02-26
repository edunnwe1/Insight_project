import pandas as pd
rom os import listdir
import glob
import sys
sys.path.append('../scripts/')
from nuclei_segmentation_opencv import nuclei_segmenter

def extract_nucleus_features(data_path,area=500,m_m=2.8,solidity=0.4):
    """
        This function extracts the nucleus features from cropped images in data_path, using nuclei_segmenter
        
        Inputs:
            data_path: containing folder of images
            area: minimum area for detected nuclei
            m_m: maximum ratio of the maximum axis length to the minimum axis length
            solidity: minimum solidity for detected nuclei
            
        Outputs:
            sipakmed_df_ext: dataframe with extracted nucleus features from all files
    """
    seg = nuclei_segmenter(cropped_flag=1,area_threshold=area,major_minor=m_m,solidity_threshold=solidity)
    
    sipakmed_df_ext = pd.DataFrame()
    folders = listdir(data_path)[1:]
    for folder in folders:
        for blob in glob.glob(data_path+'{}/CROPPED/*.bmp'.format(folder)):
            seg.read_img(blob) # read the image
            df = seg.segment_nuclei() # segment the nuclei
            if not df.empty:
                df['cluster_id'] = int(blob.split('/')[-1][0:3]) # cluster id is the first three numbers of the file name
                df['Class'] = folder[3].lower() # class id is the first letter of the containing folder
                if df['Class'] in ['s','p']:
                    df['Normal']=1
                else:
                    df['Normal']=0
                sipakmed_df_ext = sipakmed_df_ext.append(df)
    sipakmed_df_ext.to_csv('../data/extracted/Sipakmed_nuclei_ext.csv')
    return sipakmed_df_ext