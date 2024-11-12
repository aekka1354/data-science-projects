#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyforest


# In[4]:


data=pd.read_csv(r'C:\Users\aekka\Desktop\sql\diabetes.csv')


# In[7]:


data


# In[8]:


data.describe()


# In[23]:


data['BloodPressure']= data['BloodPressure'].replace(0,data['BloodPressure'].mean())
data['Glucose']= data['Glucose'].replace(0,data['Glucose'].mean())
data['Insulin']= data['Insulin'].replace(0,data['Insulin'].median())
data['SkinThickness']= data['SkinThickness'].replace(0,data['SkinThickness'].mean())
data['BMI']= data ['BMI'].replace(0,data['BMI'].mean())


# In[24]:


data


# In[25]:


fig,ax=plt.subplots(figsize=(15,10))
sns.boxplot(data=data,ax=ax,fliersize=3)


# In[28]:


q3 = data['Insulin'].quantile(0.75)    #to find outlier we have to find uppper band any value
                                       # over ub is outlier (ub = q3 + 1.5*iqr).
q1 = data['Insulin'].quantile(0.25)
iqr = q3-q1


# In[29]:


ub = q3 + 1.5 * iqr


# In[30]:


ub


# In[31]:


ub = data['Insulin'].quantile(0.95)


# In[33]:


data[data['Insulin']<ub]


# In[ ]:





# In[58]:


from sklearn.preprocessing import StandardScaler


# In[59]:


x=data.drop(['Outcome'],axis=1)


# In[60]:


y= data['Outcome']


# In[61]:


x


# In[62]:


y


# In[67]:


scalar = StandardScaler()   #standard normal distributon. mean 0 and standard deviation 1


# In[68]:


x_scaled = scalar.fit_transform(x)


# In[69]:


x_scaled


# In[ ]:





# In[78]:


from sklearn.model_selection import train_test_split


# In[79]:


x_train,x_test,y_train,y_test = train_test_split(x_scaled,y)


# In[80]:


from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


# In[81]:


from sklearn.linear_model import LogisticRegression


# In[82]:


model = LogisticRegression()


# In[84]:


model.fit(x_train,y_train)


# In[94]:


predict=model.predict(x_test)


# In[95]:


actual= y_test


# In[96]:


confusion_matrix(actual,predict)


# In[97]:


# now here the confusion matrix works
cm=confusion_matrix(actual,predict)


# In[117]:


tp = cm[0][0]
fn = cm[0][1]
fp = cm[1][0]
tn = cm[1][1]


# In[118]:


accuracy = (tp+tn)/(tp+tn+fp+fn)


# In[119]:


accuracy


# In[120]:


accuracy_score(actual,predict)


# In[126]:


precision_score(actual,predict)


# In[127]:


tp/(tp+fp)


# In[ ]:




