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






