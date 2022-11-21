# ECS171-SpamFiltering
ECS 171 Machine Learning final project.

## Introduction
Our goal is to classify messages from the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) as “ham” or “spam” through natural language processing and classification. By preprocessing and analyzing the text data, we hope to identify common indicators of malicious messages to then train a classifier on. 
Jupyter notebook viewable [here](https://colab.research.google.com/drive/14M0wI-DdfOWu4kVmE7BmxnN31tI02_9R) on Google Colab. 

## Preprocessing the Data
Several conventions exist for preprocessing text data, such as formatting the text data to be all lower-case without punctuation, emoticons, or special characters—unless the presence of these characteristics are found to be relevant to the classification of spam. It is also beneficial to strip the data of stop words which are incredibly common or considered insignificant in this context. Some common stop words are “the”, “in”, “and”, “what” and other words which appear commonly or don’t hold particular association with the categories that we aim to classify our data phrases into. Checking for null values and length of the message are some of the other things that has been done. Additionally, punctuation is removed and all alphabets are in lowercase. This is done to establish a common base to check and compare the strings. 

## Setup
Ham and Spam are respectivly encoded as 0 and 1 under column label_num.
The Jupyter Notebook and data can be downloaded from this repository and run in Google Colab.

## Project status
In development. At data exploration and preprocessing stages.

## Credits
| Authors               |
| ----------------------| 
| Atharav Ganesh Samant |
| Brinda Puri           |
| Caroline Hopkins      |
| Naomi Prem Lim        |
| Vinh Tran             |
