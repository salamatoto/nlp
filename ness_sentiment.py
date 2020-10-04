import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import urllib, json

import os
'''dir_path = 'F:\\'
os.chdir(dir_path)
'''
doc_txt = pd.read_csv('yelp_clean.csv', index_col=0)
print(doc_txt)

print(doc_txt.shape)

count = doc_txt.isnull().sum().sort_values(ascending=False)
percentage = ((doc_txt.isnull().sum() / len(doc_txt)* 100)).sort_values(ascending=False)
missing_data = pd.concat([count, percentage], axis=1, keys=['Count','Percentage'])
print('This Percentage  of missing values for the columns ...')
print(missing_data)

# to show and Distrbution of defoult

print(round(doc_txt.target.value_counts(normalize=True)* 100, 2))
round(doc_txt.target.value_counts(normalize=True)*100,2).plot(kind='bar')
plt.title('Percentage Distrutions by revires type')
plt.show()


#to convert function to lower case remove squre bracket , remove  number and punctuotion
import re
import string

def text_clean(text):
    text = text.lower()
    text = re.sub('\[.*?\']', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation),'',text)
    text = re.sub('\w*\d\w', '', text)
    return text
clean1 = lambda x: text_clean(x)


# look os take a look at the update text

doc_txt['feature_clean']  = pd.DataFrame(doc_txt.feature.apply(clean1))
print(doc_txt.head())
    

# Apply a second round od cleaing 

def text_clean1(text1):
    text1 = re.sub('[''""-]', '', text1)
    text1 = re.sub('\n','', text1)
    return text1
clean2 = lambda x: text_clean1(x)


doc_txt['feature_clean_new']  = pd.DataFrame(doc_txt.feature_clean.apply(clean2))
print(doc_txt.head())
from sklearn.model_selection import train_test_split
iv_train, iv_test, dv_train, dv_test = train_test_split(doc_txt.feature_clean_new, doc_txt.target, test_size=0.25, random_state=225)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score,accuracy_score
from sklearn.pipeline import Pipeline

tvec = TfidfVectorizer()
clf2 = LogisticRegression(solver='lbfgs')

model = Pipeline([('Vectorizer',tvec),('classifier', clf2)])
model.fit(iv_train, dv_train)

prediction = model.predict(iv_test)
print(f'Accuracy :--> {accuracy_score(prediction, dv_test)}')
print(f'Confusion :--> {confusion_matrix(prediction, dv_test)}')
print(f'Percision :--> {precision_score(prediction, dv_test)}')
print(f'Recall :-->{recall_score(prediction, dv_test)}')


# to check new data in model

exmple = ["i love take time in round the mountian"]

n = model.predict(exmple)
print(f' my answer:--> {n}')
