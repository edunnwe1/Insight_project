# Insight_project

NOTE: This project is in progress. See the 'To-do' list at the end of this readme. 

NOTE: Please also note that there was a massive data leakage issue with the dataset used for this project, which was discovered late in the game. Below, I lay out the motivation of the project, as well as the approach, to give context to the code that is written here. However, I will add a section at the end discussing the data leakage issue and why these results are at fault. This is really important for anyone planning on using this dataset in the future, as there are few publically available cervical cancer datasets out there. 

Cervical cancer is the 4th leading cancer in women, causing roughly 300,000 deaths annually, even though it is preventable with regular screening. Low income countries have the highest incidences of cervical cancer related deaths, and cervical cancer is the leading cause of death in some countries in Africa. A substantial contributing factor is that low-income countries have a shortage of trained cytotechnicians that can read cervical cancer screens (called pap smears), so that the few technicians are overburdened. 

The expertise of a cytotechnician can never truly be replaced, and as it is, cytotechnicians are able to screen for cervical cancer with reasonable accuracy (85% or above). However, searching for cancerous or precancerous cells is a needle-in-the-haystack search process. Thus, a tool that can prioritize the cytotechnician's workflow to begin with views that have the highest probability of a cancerous cell would massively improve the efficiency of a diagnosis to the patients that most need care. 

Some automated systems exist towards this end, but they are expensive, and require slide preparations to be perfectly clean with no overlap between cells. 

This project segments the nuclei of cervical cancer cells from the Sipakmed dataset (available here:https://www.cs.uoi.gr/~marina/sipakmed.html), extracts nucleus features, and classifies the cells as abnormal or normal using a random forest classifier based on those features. The reason that this project focuses on nuclei segmentation is because nuclei are less likely to be overlapping than cytoplasm, and are therefore easier to segment in thick slide preparations. A tool is created using a streamlit app that takes a cluster image of cells and identifies which cells are classified as abnormal. Future iterations will take a series of cluster images and present them rank-ordered according to the probability of containing an abnormal cell. 

The pipeline for that approach is as follows:
1. A cluster image is uploaded
2. The image is passed through 'nuclei_segmentation_streamlit'. Nuclei_segmentation_streamlit instantiates a class nuclei_segmenter, which has attributes area_threshold, P2A_threshold, and solidity_threshold, which allow rejection of detected nuclei with area, perimeter-to-area, or solitidy outside of reasonable bounds. This helps filter out debris. Additionally, nuclei_segmenter has attributes which determine the parameters for image pre-processing, including the kernel size and clipLimit for contrast-local adaptive-histogram equalization. Finally, nuclei_segmenter has attributes specificying whether or not the input image is a cluster of cells or a single-cell crop. 
3. Nuclei_segmenter has a method segment_nuclei, which first pre-processes the image with a contrast-local adaptive-histogram equalization (via another class called img_preProcessor, which can alternatively or additionally be used for median filtering). It then thresholds the image with a multiotsu threshold from skimage, binarizes the image, and extracts the nuclei. From there it determines the features of the nuclei using the regionprops method of skimage's measure. Finally, it returns a dataframe of nucleus features and the filtered image. 
4. The extracted nuclei data frame is then passed through the trained classifier (trained on ground truth, with nuclei marked by hand - details on how I trained and validated that model in a bit).
5. Finally, the original image is annotated to have bounding boxes surrounding the extracted nuclei, where a red bounding box indicates a nucleus identified as abnormal and a green bounding box indicates a nucleus identified as normal

To do, general:
- Add description of classifier and training to readme
- Clean up notebooks demonstrating exploratory data analyses and different image pre-processing techniques so that they are useful references
- Add images to illustrate the process
- Add a description and illustration of the data leakage problem, mention CNNs
- Show analyses and results for / discuss Herlev dataset

To do, features (if there weren't a data leakage problem...):
- Perhaps incorporate the image processing step as its own method in the nuclei_segmentor class
- Add a grid search over classifier parameters
- Debris removal 
- Cluster images upload in batch, rank order according to probability of containing an abnormal cell
- Improve validation method for nuclei detection 
