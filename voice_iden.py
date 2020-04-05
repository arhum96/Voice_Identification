#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# In[ ]:





# In[2]:


df = pd.read_csv("/Users/arhumsolatwaheed/Desktop/voice.csv")
df.head()


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


print(f"Shape of the data: {df.shape}")
print(f"Total number of labels {df.shape[0]}")
print(f"Number of male {df[df.label == 'male'].shape[0]}")
print(f"Number of female {df[df.label == 'female'].shape[0]}")


# In[6]:


from sklearn.preprocessing import LabelEncoder


# In[7]:


X = df.iloc[:,:-1]
y = df.iloc[:,-1]


# In[8]:


gender_encoder = LabelEncoder()
y = gender_encoder.fit_transform(y)
y


# In[9]:


from sklearn.preprocessing import StandardScaler


# In[10]:


scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 100)


# In[11]:


from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
svc_model = SVC()
svc_model.fit(X_train,y_train)
y_pred = svc_model.predict(X_test)


# In[12]:


print("Accuracy Score:")
print(metrics.accuracy_score(y_test,y_pred))


# In[13]:


print(confusion_matrix(y_test,y_pred))


# In[14]:


print(classification_report(y_test,y_pred))


# In[15]:


from sklearn.model_selection import GridSearchCV


# In[16]:


param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001]}


# In[17]:


grid = GridSearchCV(SVC(), param_grid,refit = True, verbose=2)
grid.fit(X_train, y_train)


# In[18]:


grid_predictions = grid.predict(X_test)


# In[19]:


print(f"Accuracy Score: {metrics.accuracy_score(y_test, grid_predictions)}")


# In[22]:


print(f"Changing from a SVC model to a GridSearch model improved accuracy  by {100 *(metrics.accuracy_score(y_test, grid_predictions) -metrics.accuracy_score(y_test,y_pred))}%")


# In[ ]:




