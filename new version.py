#!/usr/bin/env python
# coding: utf-8

# ## approach one

# In[1]:


#import
import nltk
import numpy as np
import pandas as pd
from sklearn import tree
from pandas import ExcelFile
from pandas import ExcelWriter
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


#dataset format
df = pd.read_excel('quora_dataset.xlsx', sheet_name='Sheet1')
print(df.columns)
df


# ## proposed system using tfidf 

# In[41]:


duplicates=pd.DataFrame(df[df.is_duplicate==1])
non_duplicates=pd.DataFrame(df[df.is_duplicate==0])
x=duplicates[:149263]
x=x.append(non_duplicates[:149263],ignore_index=True)
x=shuffle(x)


# In[45]:


import scipy
x_train=x[:100000]
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(pd.concat((x_train['question1'],x_train['question2'])).unique())
trainq1_trans = tfidf_vect.transform(x_train['question1'].values)
trainq2_trans = tfidf_vect.transform(x_train['question2'].values)
labels = x_train['is_duplicate'].values
X = scipy.sparse.hstack((trainq1_trans,trainq2_trans))
y = labels
X_train,X_valid,y_train,y_valid = train_test_split(X,y, test_size = 0.33, random_state = 42)


# In[46]:


clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X_train, y_train)
predic=clf.predict(X_valid)
print(accuracy_score(y_valid,predic))

