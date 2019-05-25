# Twitter-Rumour-Detection-
A Twitter rumour detection system using machine learning based on PHEME dataset

## Files contained in this repository:
1. feature_extraction.py<br>
Extract feature matrix from raw twitter dataset and store in a csv file "dataset.csv"
    
2. scikit-learn.py<br>
Read "dataset.csv", apply machine learning models and report the performance of testing set, plot confusion matrix
    
3. Gaussain Naive Bayes.py<br>
A Gaussian Naive Bayes model that read "dataset.csv" and implement training, testing and report performance.
    
4. dataset.csv<br>
Extracted features from dataset. The outcome of feature_extraction.py


## Software requirement:
Python 3.0 or higher version

## Library requirement: 
pandas seaborn matplotlib sklearn nltk json

## Library installation (based on Windows):
1. In windows command prompt<br>
2. Navigate to the location of the pip folder<br>
3. Enter command to install NLTK<br><br>
`pip3 install nltk`<br><br>
4. Installation should be done successfully<br>
5. Other libraries have similiar way for installation<br>
         
## PHEME Dataset Download is available at:
https://figshare.com/articles/PHEME_dataset_of_rumours_and_non-rumours/4010619

## Usage:

1. Download PHEME dataset and unzip the file, put the file together with scripts and add them into current folder in Python.<br>
2. Install relevant libraries through command prompt or Python platform.<br>
3. Specify the dataset that desired to analyse in feature_extraction.py and run feature_extraction.py.<br> 
4. Run scikit-learn.py or Gaussain Naive Bayes.py to implement classification and get results.<br>
