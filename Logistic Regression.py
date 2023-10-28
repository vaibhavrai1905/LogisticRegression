#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os #connecting local machine to your python
import numpy as np #calculation
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# ### ML >  supervised ML > Classification problem > Algorithms > Logistic Regression

# In[3]:


df1=pd.read_csv(r"C:\Vaibhav\IMARTICUS\Logistic Regression\Data.csv")


# In[21]:


df1.head(5)


# ### EDA
EDA mein hum dekhte hainki kya kya horaha hai.
# In[17]:


sns.histplot(data=df1, x="age")
plt.show()


# In[18]:


def mynum_univariate(data, x, hist_plot=True, box_plot=True):
    minn=data[x].min()
    maxx=data[x].max()
    meann=data[x].mean()
    stdd=data[x].std()
    if hist_plot==True:
        sns.histplot(data=data, x=x)
        plt.show()
    if box_plot==True:
        sns.boxplot(data=data, y=x)
        plt.show()
    print("min:" , minn, "max:", maxx, "mean:", meann, "std:", stdd)
    


# In[24]:


mynum_univariate(data=df1, x="age", hist_plot=False, box_plot=True)


# In[25]:


mynum_univariate(data=df1, x="ed", hist_plot=False, box_plot=True)


# In[26]:


mynum_univariate(data=df1, x="employ", hist_plot=False, box_plot=True)

two types of outlier
1.outlier
2.natural outlier
# In[27]:


mynum_univariate(data=df1, x="income", hist_plot=False, box_plot=True)


# In[ ]:


mynum_univariate(data=df1, x="income", hist_plot=False, box_plot=True)


# In[29]:


df2=df1.copy()
df2["income_log"]=np.log(df1["income"])


# In[32]:


mynum_univariate(data=df2, x="income_log", hist_plot=False, box_plot=True)


# ## Feature Engineering
iss mein hum missing values ko dekhte hai
# ### Missing Values

# In[33]:


df1.isnull().sum()


# ### Outlier treatment(capping)

# In[34]:


df1.describe(percentiles=[0.01,.02,.03,.04,.05,.1,.5,.75,.9,.95,.96,.97,.98,.99]).T


# In[35]:


df1.shape


# In[36]:


df1.dtypes


# In[37]:


df1.info()


# In[ ]:





# In[45]:


def myoutliers_percentile(x):
    x=x.clip(upper=x.quantile(.99))
    x=x.clip(lower=x.quantile(.01))
    return x


# In[46]:


df2=df1.apply(myoutliers_percentile)


# ### Multicollinearity

# In[50]:


cr=df2.corr()
sns.heatmap(cr,annot=True,cmap="coolwarm")
plt.show()


# ###  model development

# In[53]:


df2.columns


# In[51]:


# separate y and x
# split data into (x_train, y_train,) and (x_test, y_test)
# creat model object
# fit model on train data


# In[55]:


y=df2["default"]
#x=df2[['age', 'ed', 'employ', 'address', 'income', 'debtinc', 'creddebt', 'othdebt', 'default']]
x=df2.drop(columns=["default"])


# In[58]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.25, random_state=0)


# In[62]:


logistic_model=LogisticRegression(max_iter=10000)


# In[65]:


# help(logistic_model)


# In[1]:


logistic_model.fit(x_train,y_train)


# ### model validation

# In[68]:


logistic_model.coef_


# In[66]:


logistic_model.intercept_


# In[72]:


logistic_model.score(x_train, y_train) #Accuracy


# In[73]:


logistic_model.score(x_test, y_test) # Accuracy


# In[80]:


pd.DataFrame(logistic_model.predict_proba(x_train))


# In[ ]:





# In[76]:


pred_train


# In[77]:


pred_test=logistic_model.predict(x_test)


# In[78]:


pred_test


# In[79]:


from sklearn import metrics


# In[86]:


cmtrain=metrics.confusion_matrix(y_train, pred_train)


# In[89]:


pd.DataFrame(cmtrain, columns=["Pred_0", "Pred_1"], index=["Act_0", "Act_1"])


# In[93]:


# recall=tp/(tp+fn)
recall=77/(77+62)
recall


# In[91]:


print(metrics.classification_report(y_train, pred_train))


# In[94]:


macro_avg(.85+.70)/2


# In[96]:


weighted_avg=(.85*386+.70*139)/(386+139)


# In[97]:


weighted_avg


# In[99]:


cmtest=metrics.confusion_matrix(y_test, pred_test)
pd.DataFrame(cmtest, columns=["Pred_0", "Pred_1"], index=["Act_0", "Act_1"])


# In[100]:


print(metrics.classification_report(y_test, pred_test))


# ### chnaging the value to maximise the recall value

# In[102]:


prob_train = pd.DataFrame(logistic_model.predict_proba(x_train), columns=["prob+0","prob_1"])
prob_train.head()


# In[106]:


new_pred_train=np.where(prob_train["prob_1"]>=.3,1,0)
print(metrics.classification_report(y_train, new_pred_train))


# ### ROC : Receiver Operators Characteristics

# In[118]:


fpr, tpr,_=metrics.roc_curve(y_train, prob_train["prob_1"])


# In[119]:


plt.plot(fpr, tpr)
plt.plot([0,1],[0,1])
plt.ylabel("True positive Rate")
plt.xlabel("False Positive Rate")
plt.show()


# In[120]:


prob_test = pd.DataFrame(logistic_model.predict_proba(x_test), columns=["prob+0","prob_1"])


# ### ROC : Receiver Operators Characteristics

# In[122]:


fpr1, tpr1,_=metrics.roc_curve(y_test, prob_test["prob_1"])
auc1=metrics.auc(fpr1,tpr1)
auc1


# In[123]:


plt.plot(fpr1, tpr1)
plt.plot([0,1],[0,1])
plt.ylabel("True positive Rate")
plt.xlabel("False Positive Rate")
plt.show()


# In[124]:


new_data=pd.read_csv(r"C:\Vaibhav\IMARTICUS\Logistic Regression\cust_new.csv")


# In[126]:


new_data.head(2)


# In[127]:


new_data.drop(columns=["Unnamed: 0", "default"], inplace=True)


# In[128]:


logistic_model.predict(new_data)


# In[129]:


x_train.head()


# In[133]:


dt=pd.DataFrame(logistic_model.predict_proba(new_data), columns=["Prob_0", "prob_1"])


# In[134]:


new_pred=np.where(dt["prob_1"]>=.3,1,0)
new_pred


# In[ ]:




