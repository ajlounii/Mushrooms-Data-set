#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[4]:


orig = pd.read_csv('mushrooms.csv')


# In[5]:


orig.head()


# In[6]:


#The 'class' column contains our labels.
#It tells us whether the mushroom is 'poisonous' or 'edible'.
X = orig.drop(['class'], axis=1)
y = orig['class']


# In[7]:


for attr in X.columns:
    print('\n*', attr, '*')
    print(X[attr].value_counts())


# In[8]:


X.drop(['veil-type'], axis=1, inplace=True)


# In[9]:


for attr in X.columns:
    #Format subplots
    fig, ax = plt.subplots(1,2)
    plt.subplots_adjust(right=2)
    
    #Construct values to count in each column
    a=set(X[X['stalk-root']=='?'][attr])
    b=set(X[X['stalk-root']!='?'][attr])
    c = a.union(b)
    c = np.sort(np.array(list(c)))
    
    #Build each subplot
    sns.countplot(x=X[X['stalk-root']=='?'][attr], order=c, ax=ax[0]).set_title('stalk-root == ?')
    sns.countplot(x=X[X['stalk-root']!='?'][attr], order=c, ax=ax[1]).set_title('stalk-root != ?')
    
    #Plot the plots
    fig.show()


# In[10]:


print( (len(X[X['stalk-root']=='?']) / len(X))*100, '%', sep='') 


# In[11]:


#For columns with only two values
for col in X.columns:
    if len(X[col].value_counts()) == 2:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])


# In[12]:


X.head()


# In[13]:


X = pd.get_dummies(X)


# In[14]:


X.head()


# In[15]:


#Initialize the model
kmeans = KMeans(n_clusters=2)


# In[16]:


#Fit our model on the X dataset
kmeans.fit(X)


# In[17]:


#Calculate which mushrooms fall into which clusters
clusters = kmeans.predict(X)


# In[18]:


#'cluster_df' will be used as a DataFrame
#to assist in the visualization
cluster_df = pd.DataFrame()

cluster_df['cluster'] = clusters
cluster_df['class'] = y


# In[19]:


sns.factorplot(col='cluster', y=None, x='class', data=cluster_df, kind='count', order=['p','e'], palette=(["#7d069b","#069b15"]))


# In[20]:


kmeans = KMeans(n_clusters=3)
kmeans.fit(X)


# In[21]:


clusters = kmeans.predict(X)


# In[22]:


cluster_df = pd.DataFrame()

cluster_df['cluster'] = clusters
cluster_df['class'] = y


# In[23]:


sns.factorplot(col='cluster', y=None, x='class', data=cluster_df, kind='count', order=['p','e'], palette=(["#7d069b","#069b15"]))


# In[24]:


kmeans = KMeans(n_clusters=2)
kmeans.fit(X)


# In[25]:


clusters = kmeans.predict(X)


# In[26]:


cluster_df = pd.DataFrame()

cluster_df['cluster'] = clusters
cluster_df['class'] = y


# In[27]:


sns.factorplot(col='cluster', y=None, x='class', data=cluster_df, kind='count', order=['p','e'], palette=(["#7d069b","#069b15"]))


# In[28]:


le = LabelEncoder()
y = le.fit_transform(y)

y


# In[29]:


#Our training set will hold 80% of the data
#and the test set will hold 20% of the data
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20)


# In[30]:


#K-Means Clustering with two clusters
kmeans = KMeans(n_clusters=2)

#Logistic Regression with no special parameters
logreg = LogisticRegression()


# In[31]:


kmeans.fit(train_X)#Note that kmeans is unlabeled...

logreg.fit(train_X, train_y)#... while logreg IS labeled


# In[32]:


kmeans_pred = kmeans.predict(test_X)

logreg_pred = logreg.predict(test_X)


# In[33]:


kmeans_pred_2 = []
for x in kmeans_pred:
    if x == 1:
        kmeans_pred_2.append(0)
    elif x == 0:
        kmeans_pred_2.append(1)
        
kmeans_pred_2 = np.array(kmeans_pred_2)


# In[34]:


if accuracy_score(kmeans_pred, test_y, normalize=False) < accuracy_score(kmeans_pred_2, test_y, normalize=False):
    kmeans_pred = kmeans_pred_2


# In[35]:


#This DataFrame will allow us to visualize our results.
result_df = pd.DataFrame()

#The column containing the correct class for each mushroom in the test set, 'test_y'.
result_df['test_y'] = np.array(test_y) #(don't wanna make that mistake again!)

#The predictions made by K-Means on the test set, 'test_X'.
result_df['kmeans_pred'] = kmeans_pred
#The column below will tell us whether each prediction made by our K-Means model was correct.
result_df['kmeans_correct'] = result_df['kmeans_pred'] == result_df['test_y']

#The predictions made by Logistic Regression on the test set, 'test_X'.
result_df['logreg_pred'] = logreg_pred
#The column below will tell us whether each prediction made by our Logistic Regression model was correct.
result_df['logreg_correct'] = result_df['logreg_pred'] == result_df['test_y']


# In[36]:


fig, ax = plt.subplots(1,2)
plt.subplots_adjust(right=2)
sns.countplot(x=result_df['kmeans_correct'], order=[True,False], ax=ax[0]).set_title('K-Means Clustering')
sns.countplot(x=result_df['logreg_correct'], order=[True,False], ax=ax[1]).set_title('Logistic Regression')
fig.show()


# In[37]:


from sklearn import metrics


# In[38]:


cm = metrics.confusion_matrix(test_y, logreg_pred)
print(cm)


# In[39]:


plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(logreg_pred)
plt.title(all_sample_title, size = 15);


# In[ ]:




