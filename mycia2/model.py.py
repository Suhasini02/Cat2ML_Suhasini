#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re


# In[2]:


data=pd.read_csv("C:/Users/suhas/Downloads/datasets/final_perfume_data.csv",encoding='unicode_escape')


# In[3]:


data.head()


# In[4]:


data.drop(columns=['Image URL'],inplace=True)
data.head()


# In[5]:



data['Notes'].fillna('',inplace=True)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_mat = tfidf.fit_transform(data['Notes'])
tfidf_mat


# In[6]:


from sklearn.metrics.pairwise import linear_kernel

cosine_sim = linear_kernel(tfidf_mat,tfidf_mat)
cosine_sim


# In[7]:


data1 = pd.Series(data['Notes'],index = data.index)
data1 = pd.DataFrame(data1)
data1


# In[20]:


def recommendation(keyword):
    index = data1[data1['Notes'].str.contains(keyword, flags=re.IGNORECASE, regex=True)].index[0]
    sim_score = list(enumerate(cosine_sim[index]))    
    sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)
    
    sim_score = sim_score[1:8]
    final_index = [i[0] for i in sim_score]
    return final_index


# In[22]:


rec=str(input("Enter a flavour for recommendation : "))
idx=recommendation(rec)

print("\nRecommended Perfumes are : \n")
for i in idx:
    print('--> ',data['Name'][i])


# In[23]:


rating=pd.read_csv('C:/Users/suhas/Downloads/archive (3)/Netflix_Dataset_Rating.csv')
movie=pd.read_csv('C:/Users/suhas/Downloads/archive (3)/Netflix_Dataset_Movie.csv')
rating.head()


# In[24]:


movie.head()


# In[25]:


movie.info()


# In[34]:


users=len(rating['User_ID'].unique())
n_movies=len(rating['Movie_ID'].unique())


# In[35]:


for i in rating.itertuples():
    print(i)


# In[41]:


mat= np.zeros((users,n_movies))
for i in rating.itertuples():
    mat[i[2]-1,i[1]-1]=i[3]
mat


# In[42]:


mat[438,0]


# In[43]:


from scipy.sparse import csr_matrix
sparse_mat=csr_matrix(mat)


# In[44]:


from sklearn.neighbors import NearestNeighbors
knn=NearestNeighbors( n_neighbors=4, algorithm='brute', metric='cosine', n_jobs=-1)
knn.fit(sparse_mat)


# In[48]:


data = rating.sort_values(['User_ID'], ascending=True)
coll_filter = data[data['User_ID'] == 7].Movie_ID
coll_filter = coll_filter.tolist()
len(coll_filter)


# In[49]:


item_id=[]
for i in coll_filter:
    distances , indices = knn.kneighbors(sparse_mat[i],n_neighbors=4)
    indices = indices.flatten() #converting nested list to a single list
    indices= indices[:1]
    item_id.extend(indices)
item_id=item_id[:7] 


# In[54]:


print("Recommened Movies :\n")
for i in item_id:
    print('-->',movie['Name'][i-1], "\033[1m written by \033[0m",movie['Year'][i-1])


# In[ ]:




