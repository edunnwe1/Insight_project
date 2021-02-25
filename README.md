# Insight_project

#### Problem Statement:
Cervical cancer is the 4th leading cancer in women, causing roughly 300,000 deaths annually, even though it is preventable with regular screening. Low income countries have the highest incidences of cervical cancer related deaths, and cervical cancer is the leading cause of death in women in some of these countries, particularly in Africa. One significant challenge faced by low-income countries is a shortage of trained cytotechnicians that can read cervical cancer screens. This means that cytotechnicians are over-worked, which can lead to errors. What's more, there is often a backlog of screens, with patients not hearing results for a month or more.

#### Project Motivation: 
The expertise of a cytotechnician can never truly be replaced, and as it is, cytotechnicians are able to screen for cervical cancer with good accuracy (85% or above). However, searching for cancerous or precancerous cells is a needle-in-the-haystack search process, where technicians need to screen thousands of images. Thus, a tool that can prioritize the cytotechnician's workflow so that they review with images with the highest probability of a cancerous cell first would massively improve their efficiency. Although some automated systems exist towards this end, they are expensive, and require slide preparations to be perfectly clean with no overlap between cells. 

#### Project Value: 
This project segments the nuclei of cervical cancer cells from the Sipakmed dataset (available here:https://www.cs.uoi.gr/~marina/sipakmed.html), extracts nucleus features, and classifies the cells as abnormal or normal using a random forest classifier based on those features. The reason that this project focuses on nuclei segmentation is because nuclei are less likely to be overlapping than cytoplasm, and are therefore easier to segment in thick slide preparations. A tool is created using a streamlit app that takes a cluster image of cells and identifies which cells are classified as abnormal. Future iterations will take a series of cluster images and present them rank-ordered according to the probability of containing an abnormal cell. 

#### Code and notebooks overview:

#### Tool Description:
This tool works as follow:
1. A cluster image is uploaded
2. The image is passed through 'nuclei_segmentation_streamlit'. Nuclei_segmentation_streamlit instantiates a class nuclei_segmenter, which has attributes area_threshold, major_to_minor, and solidity_threshold, which allow rejection of detected nuclei with area, major:minor axis length ratio, or solitidy outside of reasonable bounds. This helps filter out debris. Additionally, nuclei_segmenter has attributes which determine the parameters for image pre-processing, including the kernel size and clipLimit for contrast-local adaptive-histogram equalization. Finally, nuclei_segmenter has attributes specificying whether or not the input image is a cluster of cells or a single-cell crop. 
3. Nuclei_segmenter has a method segment_nuclei, which first pre-processes the image with a contrast-local adaptive-histogram equalization (via another class called img_preProcessor, which can alternatively or additionally be used for median filtering). It then thresholds the image with a multiotsu threshold from skimage, binarizes the image, and extracts the nuclei. From there it determines the features of the nuclei using the regionprops method of skimage's measure. Finally, it returns a dataframe of nucleus features and the filtered image. 
4. The extracted nuclei data frame is then passed through the trained classifier to predict whether or not any cells are abnormal
5. Finally, the original image is annotated to have bounding boxes surrounding the extracted nuclei, where a red bounding box indicates a nucleus identified as abnormal and a green bounding box indicates a nucleus identified as normal

#### References

