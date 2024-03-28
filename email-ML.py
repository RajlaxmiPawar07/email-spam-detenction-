#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd


# In[5]:


data=pd.read_csv('spam.csv')
data


# In[6]:


data.info()


# In[7]:


data.isna().sum()


# In[8]:


data.columns


# In[9]:


data.info()


# In[14]:


data['Category']


# In[16]:


data['Spam']=data['Category'].apply(lambda x:1 if x=='spam' else 0)
data.head(5)


# In[17]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(data.Message,data.Spam,test_size=0.25)


# In[18]:


from sklearn.feature_extraction.text import CountVectorizer


# In[19]:


from sklearn.naive_bayes import MultinomialNB


# In[21]:


from sklearn.pipeline import Pipeline
clf=Pipeline([
    ('vectorizer',CountVectorizer()),
    ('nb',MultinomialNB())
])


# In[22]:


clf.fit(X_train,y_train)


# In[23]:


Pipeline(steps=[('vectorizer', CountVectorizer()), ('nb', MultinomialNB())])


# In[24]:


emails=[
    'Sounds great! Are you home now?',
    'Will u meet ur dream partner soon? Is ur career off 2 a flyng start? 2 find out free, txt HORO followed by ur star sign, e. g. HORO ARIES'
]


# In[25]:


clf.predict(emails)


# In[26]:


clf.score(X_test,y_test)


# In[ ]:




