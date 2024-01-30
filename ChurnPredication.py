#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv('Churn.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


pd.crosstab(df['PaymentMethod'],df['Churn']).plot.bar()


# In[6]:


from matplotlib import pyplot as plt


# In[7]:


a=pd.crosstab(df['Dependents'],df['Churn'])
a.plot.bar()


# In[8]:


pd.crosstab(df['Partner'],df['Churn']).plot(kind="bar",stacked=True)


# In[9]:


a=pd.crosstab(df['SeniorCitizen'],df['Churn'])
a.plot.bar()


# In[10]:


#PhoneService
a=pd.crosstab(df['PhoneService'],df['Churn'])
a.plot.bar()


# In[11]:


import seaborn as sns
sns.barplot(x ='Churn', y ='tenure', data = df, 
            palette ='plasma', estimator = np.std)


# In[12]:


#MonthlyCharges
sns.barplot(x ='Churn', y ='MonthlyCharges', data = df, 
            palette ='plasma', estimator = np.std)


# In[13]:


d=pd.crosstab(df['DeviceProtection'],df['Churn'])
d


# In[14]:


d.plot.bar()


# In[15]:


df.corr()


# In[16]:


#StreamingTV
d=pd.crosstab(df['StreamingTV'],df['Churn'])
d.plot.bar()


# In[17]:


pd.crosstab(df['PaperlessBilling'],df['Churn']).plot(kind="bar",stacked=True)


# In[18]:


#Contract 
pd.crosstab(df['Contract'],df['Churn']).plot(kind="bar",stacked=True)


# In[19]:


df['TotalCharges']=df['TotalCharges'].astype('float')


# In[20]:


sns.barplot(x='Churn',y='TotalCharges',data=df,palette ='Purples_d')


# In[21]:


df


# In[22]:


from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()


# In[23]:


def result(raw):
    return 1 if(raw['Churn']=='Yes') else 0


# In[24]:


df['Churn']=df['Churn'].replace("Yes",1)
df['Churn']=df['Churn'].replace("No",0)


# In[25]:


df


# In[26]:


#df=df.drop(columns=[
#    'StreamingTV',
#    'Dependents',
#    'PhoneService'
#])


# In[27]:


df.info()


# In[28]:


X=df[['SeniorCitizen','Partner','tenure','MultipleLines','InternetService','OnlineSecurity','DeviceProtection','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges']]
y=df['Churn']


# In[29]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)


# In[30]:


a=['Partner','MultipleLines','InternetService','OnlineSecurity','DeviceProtection','Contract','PaperlessBilling','PaymentMethod']


# In[31]:


a


# In[48]:


from sklearn.compose import ColumnTransformer


# In[49]:


ohe=OneHotEncoder()
ohe.fit([a])
trf=ColumnTransformer([
    ('trf',OneHotEncoder(max_categories=12,sparse_output=False),a)
]
,remainder='passthrough')


# In[50]:


ohe.categories_


# In[51]:


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[52]:


c1=LogisticRegression()
c2=DecisionTreeClassifier()
c3=  KNeighborsClassifier()


# In[67]:


# Number of trees in random forest
n_estimators = [20,60,100,120]

# Number of features to consider at every split
max_features= [0.2,0.6,1.0]

# Maximum number of levels in tree
max_depth = [2,8,None]

# Number of samples
max_samples = [0.5,0.75,1.0]

# Bootstrap samples
bootstrap = [True,False]

# Minimum number of samples required to split a node
min_samples_split = [2, 5]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2]
param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
              'max_samples':max_samples,
              'bootstrap':bootstrap,
              'min_samples_split':min_samples_split,
              'min_samples_leaf':min_samples_leaf
             }
print(param_grid)


# In[69]:


from sklearn.model_selection import RandomizedSearchCV

rf_grid = RandomizedSearchCV(estimator = RandomForestClassifier(), 
                       param_distributions = param_grid, 
                       cv = 5, 
                       verbose=2, 
                       n_jobs = -1)


# In[70]:


es=[('lr',c1),('knn',c2),('rf',c3)]


# In[133]:


pipe=Pipeline(steps=[
    ('step1',trf),
    ('step2',RandomForestClassifier(max_features=0.25,max_depth=0,max_samples=0.4,min_samples_split=5))
]
)


# In[134]:


pipe.fit(X_train,y_train)


# In[135]:


y_pred=pipe.predict(X_test)
y_pred


# In[136]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)


# In[86]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[77]:


X


# In[78]:


n=pipe.predict_proba(pd.DataFrame(columns=['SeniorCitizen','Partner','Dependents','tenure','PhoneService','MultipleLines','InternetService','OnlineSecurity','DeviceProtection','StreamingTV','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges'],data=np.array([0,'No','No', 6,'No','Yes','Fiber optic','No','Yes','Yes','Two year','No','Bank transfer (automatic)',1.85,64.50]).reshape(1,15))).astype(float)


# In[79]:


n[0]


# In[80]:


df.to_csv('churna.csv')


# In[81]:


373/(373+212)*100


# In[ ]:





# In[ ]:




