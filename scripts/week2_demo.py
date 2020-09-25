import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import seaborn as sns
import statistics as stats
from os import listdir
from skimage import io, filters, color,exposure,feature,measure,segmentation
from scipy import ndimage
import cv2 
import sys
sys.path.append('../scripts/')
sys.path.append('../models/')
from nuclei_segmentation_streamlit import nuclei_segmenter
import pickle
Sipakmed_model = pickle.load(open('../models/Sipakmed_nuc_model','rb'))

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("CC_Screen example")

uploaded_file = st.file_uploader("Choose an image...", type="bmp")
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    # upload the raw grayscale image
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    # segment nuclei
    seg = nuclei_segmenter()
    nuclei_df,filt_img = seg.segment_nuclei(img)
    # display the filtered image
    st.image(filt_img, caption='Filtered Image.', use_column_width=True)
    # classify the identified nuclei
    preds = Sipakmed_model.predict(nuclei_df[['area','eccentricity','solidity','meanI_R','meanI_G','meanI_B']])
    # display image with bounding boxes around abnormal nuclei
    flagged_nuclei = nuclei_df.iloc[preds==0]
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.imshow(img)
    for i in range(len(nuclei_df)):
        rect = plt.Rectangle((nuclei_df.iloc[i]['bbox-1'],nuclei_df.iloc[i]['bbox-0']),nuclei_df.iloc[i]['bbox-3']-nuclei_df.iloc[i]['bbox-1'],nuclei_df.iloc[i]['bbox-2']-nuclei_df.iloc[i]['bbox-0'],fill=False,color='g')
        ax.add_patch(rect)
    for i in range(len(flagged_nuclei)):
        rect = plt.Rectangle((nuclei_df.iloc[i]['bbox-1'],nuclei_df.iloc[i]['bbox-0']),nuclei_df.iloc[i]['bbox-3']-nuclei_df.iloc[i]['bbox-1'],nuclei_df.iloc[i]['bbox-2']-nuclei_df.iloc[i]['bbox-0'],fill=False,color='r')
        ax.add_patch(rect)
    plt.axis('off')
    st.pyplot(fig)
    st.text('%d abnormal cells detected!' % (len(flagged_nuclei)))
        
    
    