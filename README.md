# ECS171-SpamFiltering
ECS 171 Machine Learning final project.

## Introduction
Our goal is to classify messages from the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) as “ham” or “spam” through natural language processing and classification. By preprocessing and analyzing the text data, we hope to identify common indicators of malicious messages to then train a classifier on. 
Jupyter notebook viewable [here](https://colab.research.google.com/drive/14M0wI-DdfOWu4kVmE7BmxnN31tI02_9R) on Google Colab. 

## Preprocessing the Data
Several conventions exist for preprocessing text data, such as formatting the text data to be all lower-case without punctuation, emoticons, or special characters—unless the presence of these characteristics are found to be relevant to the classification of spam. It is also beneficial to strip the data of stop words which are incredibly common or considered insignificant in this context. Some common stop words are “the”, “in”, “and”, “what” and other words which appear commonly or don’t hold particular association with the categories that we aim to classify our data phrases into. Checking for null values and length of the message are some of the other things that has been done. Additionally, punctuation is removed and all alphabets are in lowercase. This is done to establish a common base to check and compare the strings. 

We began experimenting with the data by analyzing the lengths of the messages to try to find a pattern between messages in the different categories. We then tried to plotted charts for the appearance of certain words in ham or spam messages to see if certain words were more likely to appear in spam messages.

## Setup
Ham and Spam are respectivly encoded as 0 and 1 under column label_num.
The Jupyter Notebook and data can be downloaded from this repository and run in Google Colab.

## First Model
We began training our first model,and we started by processing the data and encoding the data. We then split the data into training and testing data using a 80:20 ratio, and used SVM with a Linear Kernel to train the data.

## Project status
In development. Finished with major preprocessing of data and trained first model. 


## Credits
| Authors               |
| ----------------------| 
| Atharav Ganesh Samant |
| Brinda Puri           |
| Caroline Hopkins      |
| Naomi Prem Lim        |
| Vinh Tran             |
