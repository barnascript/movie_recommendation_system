#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


movies = pd.read_csv("movies.csv")


# In[3]:


movies


# ## remove outliers to make search easier

# In[4]:


import re

def clean_title(title):
    return re.sub("[^a-zA-Z0-9 ]","", title)

movies["clean_title"] = movies["title"].apply(clean_title)


# In[5]:


movies


# ## turn text into numbers

# In[6]:


from sklearn.feature_extraction.text import TfidfVectorizer


vectorizer = TfidfVectorizer(ngram_range = (1, 2))

tfidf = vectorizer.fit_transform(movies["clean_title"])


# ## return similarity in title

# In[7]:


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def search(title):
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()

    # get the most similar movies
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies.loc[indices][:: -1]

    return results


# ## create a search box for inputing title

# In[8]:


import ipywidgets as widgets
from IPython.display import display


movie_input = widgets.Text(
    
    value = "Toy Story",
    description = "Movie Title",
    disabld = False
)

movie_list = widgets.Output()

def on_type(data):
    with movie_list:
        movie_list.clear_output()
        title = data["new"]
        if len(title) > 5:
            display(search(title))

movie_input.observe(on_type, names = "value")
display(movie_input, movie_list)


# ## finding users who liked the same movies as us and other movies they liked

# In[9]:


ratings =  pd.read_csv("ratings.csv")


# In[10]:


ratings


# In[11]:


movie_id = 1


# In[12]:


# people who like the same movies as we entered

similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()


# In[13]:


similar_users


# In[14]:


# other movies they liked

similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]


# In[15]:


similar_user_recs


# In[16]:


# get the percentage of the recommended movies

similar_user_recs = similar_user_recs.value_counts() / len(similar_users)

# only take the ones greter than 10%

similar_user_recs = similar_user_recs[similar_user_recs > .1]


# In[17]:


similar_user_recs


# In[18]:


all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]


# In[19]:


all_users_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())


# In[20]:


all_users_recs


# In[21]:


rec_percentages = pd.concat([similar_user_recs, all_users_recs],  axis=1)

rec_percentages.columns = ["similar", "all"]


# In[22]:


rec_percentages


# In[23]:


rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]


# In[24]:


rec_percentages = rec_percentages.sort_values("score", ascending = False)


# In[25]:


rec_percentages


# In[26]:


rec_percentages.head().merge(movies, left_index = True, right_on = "movieId")


# In[29]:


def find_similar_movies(movie_id):
    similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]
    
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)
    similar_user_recs = similar_user_recs[similar_user_recs > .1]
    
    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]
    all_users_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
    
    rec_percentages = pd.concat([similar_user_recs, all_users_recs],  axis=1)
    rec_percentages.columns = ["similar", "all"]
    
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending = False)
    
    return rec_percentages.head().merge(movies, left_index = True, right_on = "movieId")[["score", "title", "genres"]]


# In[30]:


movie_name_input = widgets.Text(
    
    value = "Toy Story",
    description = "Movie Title:",
    disabled = False

)

recommendation_list = widgets.Output()

def on_type(data):
    with recommendation_list:
        recommendation_list.clear_output()
        title = data["new"]
        if len(title) > 5:
            results = search(title)
            movie_id = results.iloc[0]["movieId"]
            display(find_similar_movies(movie_id))

movie_name_input.observe(on_type, names = "value")
display(movie_name_input, recommendation_list)


# In[ ]:




