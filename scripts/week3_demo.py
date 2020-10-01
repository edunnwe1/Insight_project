import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

import cv2
from torchvision import transforms
from PIL import Image

import sys
sys.path.append('../scripts/')
sys.path.append('../models/')

import pickle
pca = pickle.load(open('../models/Sipakmed_cluster_cnn3_pcaTransform','rb'))
clf = pickle.load(open('../models/Sipakmed_cluster_cnn3_pca','rb'))

from CNN_model import Resize_Alexnet
from image_preprocessing import img_preProcessor
prep = img_preProcessor()

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("CC_Screen example")


uploaded_file = st.file_uploader("Choose an image...", type="bmp")
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    master_img = cv2.imdecode(file_bytes, 1)
    # upload the raw image to web app 
    st.image(cv2.cvtColor(master_img,cv2.COLOR_BGR2RGB), caption='Uploaded Image.', use_column_width=True)
    
    #divide into sub images
    
    imgs = []
    for i,x in enumerate(np.split(master_img,[1536],axis=0)):
        for j,y in enumerate(np.split(x,[2048],axis=1)):
            imgs.append(y)
            
    # extract features
    model = Resize_Alexnet(depth=3)
    model.eval()
    imgs_df = pd.DataFrame()
    
    for img in imgs:
        # pre processing steps
        img = prep.apply_med_filt(prep.apply_CLAHE(img))
        # convert to PIL image
        img = Image.fromarray(img)
        # convert to tensor
        img_as_tensor = transforms.ToTensor()(img).unsqueeze_(0)

        # get features
        features = model(img_as_tensor).squeeze(axis=0).sum(axis=[1,2]).detach().numpy() #sum pooling, paper says good extractor

        # make df, concatenate
        df = pd.DataFrame(features).transpose()
        imgs_df = pd.concat([imgs_df,df])

    del model
    
    # reduce features via PCA
    X_pca = pca.transform(imgs_df)
    
    # predict and plot in order, according to probability of abnormal 
    probs = clf.predict_proba(X_pca[:,0:25])
    #(sorting ascending probability of being normal - top image is most abnormal) 
    for _ in np.argsort(probs[:,1]): 
        st.text('predicted abnormal with probability {}'.format(probs[_,0]))
        fig = plt.figure()
        plt.imshow(cv2.cvtColor(imgs[_], cv2.COLOR_BGR2RGB))
        plt.axis('off')
        st.pyplot(fig)



