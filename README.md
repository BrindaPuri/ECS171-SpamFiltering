# ECS171-SpamFiltering
ECS 171 Machine Learning final project.

# Introduction
Our goal is to classify messages from the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) as “ham” or “spam” through natural language processing and classification. By preprocessing and analyzing the text data, we hope to identify common indicators of malicious messages to then train a classifier on. 
Jupyter notebook viewable [here](https://colab.research.google.com/drive/14M0wI-DdfOWu4kVmE7BmxnN31tI02_9R) on Google Colab. 

# Methods
## Data Exploration
We loaded the data into a pandas dataframe df. Looking at the dataset, we realized that messages could only be classified as spam or ham (not spam). We looked at the distribution of ham and spam messages in the dataset, looked for null values that would need to be replaced or removed.
```
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
df.rename(columns={'v1':'label','v2':'message'},inplace=True)
df.groupby('label').describe()
df.isna().sum()
```


## Preprocessing the Data
We began experimenting with the data by analyzing the lengths of the messages to try to find a pattern between messages in the different categories. We stripped stopwords from the text and removed punctuation and converted the messages to lowercase.
```
def remove_punctuation(text):
  for punct in string.punctuation:
    text = text.replace(punct, " ")
  return text

def remove_contraction(text):
  contraction_endings = ['s', 't', 'd', 'll', 're', 'm']
  return " ".join([word for word in str(text).split() if word not in contraction_endings])

# Lower case
df.loc[:,'preprocess_text'] = df['message']
df.loc[:,'preprocess_text'] = df['preprocess_text'].str.lower()

# List of common stopwords to remove
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stop_words])

# Apply to df
df.loc[:,'preprocess_text'] = df['preprocess_text'].apply(lambda text: remove_punctuation(text))
df.loc[:,'preprocess_text'] = df['preprocess_text'].apply(lambda text: remove_contraction(text))
df.loc[:,'preprocess_text'] = df['preprocess_text'].apply(lambda text: remove_stopwords(text))
```
We then could look at the frequency of the top words in spam and ham messages with the preprocessed data.
```
spam_common=Counter(" ".join(spam["preprocess_text"]).split()).most_common(20)
spam_df=pd.DataFrame(spam_common)

# Pie charts for 20 most common words in spam messages
for i in range(len(spam_df)):
  new=df.loc[df['preprocess_text'].str.contains(spam_df['Words'][i])]
  spam=new.loc[new['label'] == 'spam']
  ham=new.loc[new['label'] == 'ham']
  new['label'].value_counts().plot(kind='pie')
  plt.title(spam_df['Words'][i])
  plt.show()
  percentage=len(spam)/len(new)*100
  print(f'{percentage}% of messages containing the word \"'+spam_df['Words'][i]+'\" are spam')
```


## Setup
Ham and Spam are respectively encoded as 0 and 1 under column label_num. The Jupyter Notebook and data can be downloaded from this repository and run in Google Colab.

## First Model
In order to evaluate the importance of particular words or phrases in spam classification, we used TF-IDF word vectorization to represent our messages. We split the data into training and testing data using a 80:20 ratio, and used SVM with a Linear Kernel to train the data. We evaluated the model’s performance using a confusion matrix and mean square error of the training and test set.
```
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm

X_train, X_test, y_train, y_test = train_test_split(X.values, Y.values,test_size = 0.2,random_state =3)
vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(X_train)
test_vectors = vectorizer.transform(X_test)
model = svm.SVC(kernel= 'linear')
model.fit(train_vectors,y_train)
test_pred=model.predict(test_vectors)
train_pred =model.predict(train_vectors)
```

## Second Model
Next, we used a multinomial Bayesian classifier. This model also used the TF-IDF vectorized data and the name training and testing sets. The Bayesian model was also evaluated by a confusion matrix and mean squared error.
```
from sklearn.naive_bayes import MultinomialNB

model_2 = MultinomialNB().fit(train_vectors, y_train)
test_pred = model_2.predict(test_vectors)
train_pred = model_2.predict(train_vectors)
```


# Results
Using Naive Bayes, the model classified ham messages very well, but performed poorly classifying spam messages. Many spam messages were incorrectly labeled as ham, making the first model much preferable for effectively flagging spam messages. One of the factors of this could be because the dataset has more ham messages than spam thus the model is more effectively trained to recognize ham. 

# Discussion
## Data Exploration
We looked through the data and tried to find any type of patterns that would differentiate between a spam or a ham message. Some of the first details we looked at was the length of the message and the top common words for a spam and a ham message. One pattern that we began to notice was that for spam messages, there tended to be more words involving the person receiving the message like, “you”, “your”, and “to”. Another type of words that were commonly mentioned were actions words like “call”. This pattern is further proven when we made graphs showing the usage of these certain words in spam messages and ham messages. Words like “text", "txt", "reply", "claim" were overwhelmingly common in spam messages. We also began noticing the other types of keywords that were very common which were monetary words like “free”, “prize”, “cash”.

## Preprocessing the Data
Several conventions exist for preprocessing text data, such as formatting the text data to be all lower-case without punctuation, emoticons, or special characters—unless the presence of these characteristics are found to be relevant to the classification of spam. It is also beneficial to strip the data of stop words which are incredibly common or considered insignificant in this context. Some common stop words are “the”, “in”, “and”, “what” and other words which appear commonly or don’t hold particular association with the categories that we aim to classify our data phrases into. Checking for null values and length of the message are some of the other things that have been done. Additionally, punctuation is removed and all alphabets are in lowercase. This is done to establish a common base to check and compare the strings.


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
