This project is about building NLP model that detects spam SMS text messages and because it's not fun at all to just build a model and forget about it as 90% of ML models in the world ,I'll be the exception :wink: and I'll deploy my model into an MLOps platform 
## Plan 
- create an NLP model that detects spam SMS text messages
- deploy our model into the Algorithmia platform
- use your deployed NLP model in any Python application.

# 1. Build the ML Model

### Import Python Packages
We first import all the importance python packages that we will use to load the data, preprocess the data, and create a text classification model.
```
# import important modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from string import punctuation 

# sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    plot_confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score, RandomizedSearchCV

# text preprocessing modules
from nltk.tokenize import word_tokenize
from cleantext import clean

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
import re #regular expression


from wordcloud import WordCloud, STOPWORDS

# Download dependency
for dependency in (
    "brown",
    "names",
    "wordnet",
    "averaged_perceptron_tagger",
    "universal_tagset",
    "stopwords"
):
    nltk.download(dependency)

#nltk.download('stopwords')

import warnings
warnings.filterwarnings("ignore")
# seeding
np.random.seed(123)
```
### load data
```
data = pd.read_csv("../data/spam.tsv", sep="\t")
```
### Handling Missing Values
Sometimes data can have missing values. We can use the isnull() method from pandas to check if our dataset has any missing values.
```
# check missing values
data.isnull().sum()
```
===> The output shows that our dataset does not have any missing values.(Thank God 😍)
### Evaluating Class Distribution
We can use the **value_counts()** method from the pandas package to evaluate the class distribution form our dataset.
```
# evalute class distribution
data["label"].value_counts()
```
### EDA
EDA is your way to know your data and to understand the strength of your data and what will help you to succeed your data science project.
In this step we are going to find frequent words that are used in both legitimate and spam messages.
```
# collect words from the dataset
def collect_words(data, label):
    collected_words = " "

    # iterate through the csv file
    for val in data.message[data["label"] == label]:

        # typecaste each val to string
        val = str(val)

        # split the value
        tokens = val.split()

        # Converts each token into lowercase
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()

        for words in tokens:
            collected_words = collected_words + words + " "

    return collected_words
```
This function called collect_words() will collect all words from the dataset according to their labels (ham or spam).

Then we can visualize frequent words by using the wordcloud Python package. We will start with messages labeled as ham (legitimate).
```
# visualize ham labeled sms
cloud_stopwords = set(STOPWORDS)
ham_words = collect_words(data, label="ham")

print("Total words {}".format(len(ham_words)))

wordcloud = WordCloud(
    width=1000,
    height=1000,
    background_color="white",
    stopwords=cloud_stopwords,
    min_font_size=10,
).generate(ham_words)

# plot the WordCloud image
plt.figure(figsize=(15, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)

plt.show()
```
==> We got this 'beautiful' wordCloud of the world used in legit messages😜
![wordcloud](https://github.com/inesgh1/spam_sms_detector/blob/main/word%20map.png)

==> As you can see :eyes: in the legit messages, the most frequent words are will, gt, now, ok, call, want, got, and so on.Next we'll do the same for the spam messages.
### Processing the Data
After exploring and analyzing the dataset, the next step is to preprocess the dataset into the right format before creating our machine learning model.

We first replace the ham and spam classes with numerical values. The ham class will be labeled as 0 and spam class will be labeled as 1.
```
# replace ham to 0 and spam to 1
new_data = data.replace({"ham": 0, "spam": 1})
new_data.head()
```

The messages in this dataset contain a lot of unnecessary words and characters that we don't need when creating machine learning models.

We will clean the messages by removing stopwords, numbers, and punctuation. Then we will change words into lower case, and finally convert each word into its base form by using the lemmatization process in the NLTK package.

The **text_cleaning()** function will handle all necessary steps to clean our dataset.
```
stop_words =  stopwords.words('english')

def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    # Clean the text, with the option to remove stop_words and to lemmatize word

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r"ur", " your ", text)
    text = re.sub(r" nd "," and ",text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r" tkts "," tickets ",text)
    text = re.sub(r" c "," can ",text)
    text = re.sub(r" e g ", " eg ", text)
    text =  re.sub(r'http\S+',' link ', text)
    text = re.sub(r'\b\d+(?:\.\d+)?\s+', '', text) # remove numbers
    text = re.sub(r" u "," you ",text)
    text = text.lower()  # set in lowercase 
        
    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])
    
    # Optionally, remove stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
    
    # Optionally, shorten words to their stems
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer() 
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)
    
    # Return a list of words
    return(text)
```
Now we can clean our dataset by using the text_cleaning() function.    
```    
    #clean the dataset 
new_data["clean_message"] = new_data["message"].apply(text_cleaning)
```   
Now we split our dataset into train and test data. The test size is 15% of the entire dataset.
```
# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    new_data["clean_message"],
    new_data["label"],
    test_size=0.15,
    random_state=0,
    shuffle=True,
    stratify=data["label"],
)
```
The CountVectorizer method from scikit-learn will help us transform our cleaned dataset into numerical values. The method converts a collection of text documents to a matrix of token counts.
```
# Transform text data 
vectorizer = CountVectorizer(lowercase=False)
vectorizer.fit(X_train)

#transform train data 
X_train_trans = vectorizer.transform(X_train)

#transform test data
X_text_trans = vectorizer.transform(X_test)
```
### Create Our Model
We will train the Multinomial Naive Bayes algorithm to classify if a message is legitimate or spam. This is one of the most common algorithms used for text classification.Then we train our classifier by using cross validation to avoid overfitting.
```
# Create a classifier

spam_classifier = MultinomialNB()

# Train the model with cross validation
scores = cross_val_score(spam_classifier,X_train_trans,y_train,cv=10,verbose=3,n_jobs=-1)

# find the mean of the all scores
scores.mean()
```
The mean of the scores is around 97.68%. Our model performs well, but we can improve its performance by optimizing its hyperparameter values with the Randomized Search method from scikit-learn.
```
# fine turning model parameters

distribution = {"alpha": [1, 0.1, 0.01, 0.001, 0.0001, 0, 0.2, 0.3]}

grid = RandomizedSearchCV(
    spam_classifier,
    param_distributions=distribution,
    n_jobs=-1,
    cv=10,
    n_iter=20,
    random_state=42,
    return_train_score=True,
    verbose=2,
)
```
We will optimize the alpha hyparameter from our model to get the best value that will increase our model's performance.
```
# training with randomized search
grid.fit(X_train_trans, y_train)

# summarize the results of the random parameter search
print(grid.best_score_)
print(grid.best_estimator_)
print(grid.best_params_)
```
==> The best score is the same as the previous one. Now let's test our model with the test data.

The accuracy of our model is around 97.6%, which is good performance.That's mean that our model has a good performance and now we can deploy it to production.

The model will be saved in models directory.Our  Count Vectorizer will also be saved in the preprocessing directory.
```
#save model 
import joblib 

joblib.dump(spam_classifier, '../models/spam-detection-model.pkl')

#save Vectorizer
joblib.dump(vectorizer,'../preprocessing/count_vectorizer.pkl')
```
 Congrats 🎉 :tada:  you've build your model now it's deployment time.    
  
# What is Algorithmia?
Algorithmia is a MLOps tool that provides a simple and faster way to deploy your machine learning model into production.

Algorithmia specializes in "algorithms as a service". It allows users to create code snippets that run the ML model and host them on Algorithmia. Then you can call your code as an API.

Now your model can be used for different applications of your choice, such as web apps, mobile apps, or e-commerce with a simple API call from Algorithmia.
Machine learning models created using several computer languages, including R, Python, Java, and Scala, are supported by algorithms. Additionally, it supports well-known deep learning and machine learning frameworks including Keras, Pytorch, Tensorflow, Scikit-Learn, and XGBoost.

In order to minimize costs and increase performance to meet your demands, Algorithmia's serverless Artificial Intelligence layer utilizes both CPUs and GPUs.

Currently, this platform features 4,500 algorithms and over 60,000 developers.

You must take the following six actions in order to deploy your machine learning model on Algorithmia.
    
    
    
    
    







