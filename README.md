# Insight_project

#### Problem Statement:
Cervical cancer is the 4th leading cancer in women, causing roughly 300,000 deaths annually, even though it is preventable with regular screening. 85% of cervical cancer related deaths occur in low-income countries [1], and cervical cancer is the leading cause of female cancer in roughly half of the countries in sub-saharan Africa [1]. One significant challenge faced by low-income countries is a shortage of trained cytotechnicians that can read cervical cancer screens [2]. This means that cytotechnicians are over-worked, which can lead to errors. What's more, there is often a backlog of screens, with patients not hearing results for 1-3 months [2].

#### Project Motivation: 
The expertise of a cytotechnician can never truly be replaced, and as it is, cytotechnicians are able to screen for cervical cancer with good accuracy [3]. However, searching for cancerous or precancerous cells is a needle-in-the-haystack search process, where technicians are searching for as few as 10-20 malignant cells out of over 100,000. Thus, a tool that can prioritize the cytotechnician's workflow so that they review clusters of cells with the highest probability of containing a cancerous cell first would massively improve their efficiency. Although some automated systems exist towards this end, they are expensive, and require slide preparations to be perfectly clean with no overlap between cells. This is unrealistic for most low-income countries, where an automated system would be most beneficial.

#### Project Value: 
This project segments the nuclei of cervical cancer cells from the Sipakmed dataset (available here:https://www.cs.uoi.gr/~marina/sipakmed.html), extracts nucleus features, and classifies the cells as abnormal or normal using a random forest classifier based on those features. The reason that this project focuses on nuclei segmentation is because nuclei are less likely to be overlapping than cytoplasm, and are therefore easier to segment in thick slide preparations. A tool is created using a streamlit app that takes a cluster image of cells and identifies which cells are classified as abnormal. Future iterations will take a series of cluster images and present them rank-ordered according to the probability of containing an abnormal cell. 

#### Code and notebooks overview:
##### Notebooks
There are three notebooks, ordered numerically and named with short descriptors. They illustrate exploratory data analysis (EDA), the functionality of the nucleus segmenter code, and the development and hyperparameter tuning of the model. 
##### Code (Scripts)
There are two main Python classes written for this project, the nucleus_segmenter (imported from nuclei_segmentation_opencv.py) and the image_preProcessor (imported from image_preprocessing.py). The nucleus_segmenter class is the most important. Additionally, scripts were written with helpful data wrangling functions. sipakmed_to_csv.py compiles the external .dat data into a single csv file, and extract_nucleus_features.py applies the nucleus_segmenter in batch to all cropped images in the database. Finally, streamlit_demo.py is the code to implement a local streamlit demo of the tool (described in the next section).
##### Models
Trained models from notebook #3 are stored in the models folder and used in the streamlit demo. 
##### Data
The data (not included here) should have the following folders:
* external 
* extracted
* processed

The external folder contains the sipakmed database features and pictures. The extracted folder is where extracted nucleus features are saved. The processed folder is where the compiled sipakmed database is saved. Future iterations of this project will include code to set up the appropriate file structure. 

#### Tool Description:
This tool works as follow:
1. A cluster image is uploaded
2. The image is passed through 'nuclei_segmentation_streamlit'. Nuclei_segmentation_streamlit instantiates a class nuclei_segmenter, which has attributes area_threshold, major_to_minor, and solidity_threshold, which allow rejection of detected nuclei with area, major:minor axis length ratio, or solitidy outside of reasonable bounds. This helps filter out debris. Additionally, nuclei_segmenter has attributes which determine the parameters for image pre-processing, including the kernel size and clipLimit for contrast-local adaptive-histogram equalization. Finally, nuclei_segmenter has attributes specificying whether or not the input image is a cluster of cells or a single-cell crop. 
3. Nuclei_segmenter has a method segment_nuclei, which first pre-processes the image with a contrast-local adaptive-histogram equalization (via another class called img_preProcessor, which can alternatively or additionally be used for median filtering). It then thresholds the image with a multiotsu threshold from skimage, binarizes the image, and extracts the nuclei. From there it determines the features of the nuclei using the regionprops method of skimage's measure. Finally, it returns a dataframe of nucleus features and the filtered image. 
4. The extracted nuclei data frame is then passed through the trained classifier to predict whether or not any cells are abnormal
5. Finally, the original image is annotated to have bounding boxes surrounding the extracted nuclei, where a red bounding box indicates a nucleus identified as abnormal and a green bounding box indicates a nucleus identified as normal

#### Status:
The model was successfully trained on the existing Sipakmed database of nucleus features. The code to extract the nuclei has been written and is functional, but is not yet very precise, working better on some images than others. The entire pipeline has been converted into a streamlit app.

Future iterations should allow the upload of multiple images, and the tool should arrange those images in order of probability of an abnormal cell (prob_a output from the model)

#### References
1. Estimates of incidence and mortality of cervical cancer in 2018: a worldwide analysis https://www.thelancet.com/journals/langlo/article/PIIS2214-109X(19)30482-6/fulltext#seccestitle130 
2. Challenges to Cervical Cancer in the Developing Countries: South African Context https://www.intechopen.com/books/topics-on-cervical-cancer-with-an-advocacy-for-prevention/challenges-to-cervical-cancer-in-the-developing-countries-south-african-context
3. Screening for Cervical Cancer Using Automated Analysis of PAP-Smears https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3977449/#B14
4. Africa urged to accelerate control of cervical cancer https://www.scidev.net/sub-saharan-africa/news/africa-urged-to-accelerate-control-of-cervical-cancer/
5. No woman should die from cervical cancer in Africa https://www.iaea.org/newscenter/news/no-woman-should-die-from-cervical-cancer-in-africa


