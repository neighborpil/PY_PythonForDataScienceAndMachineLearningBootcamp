#!/usr/bin/env python
# coding: utf-8

# In[1]:


import plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


# In[2]:


init_notebook_mode(connected=True)


# In[3]:


import pandas as pd


# In[4]:


data = dict(type = 'choropleth',
           locations = ['AZ', 'CA', 'NY'],
           locationmode = 'USA-states',
           colorscale = 'Portland',
           text = ['text1', 'text2', 'text3'],
           z = [1.0, 2.0, 3.0], # actual color scale
           colorbar = {'title': 'Colorbar Title'})


# In[5]:


data


# In[6]:


layout = dict(geo = {'scope': 'usa'})


# In[7]:


layout


# In[11]:


choromap = go.Figure(data = [data], layout = layout)


# In[13]:


choromap


# In[14]:


iplot(choromap)


# In[16]:


data2 = dict(type = 'choropleth',
           locations = ['AZ', 'CA', 'NY'],
           locationmode = 'USA-states',
           colorscale = 'Jet',
           text = ['text1', 'text2', 'text3'],
           z = [1.0, 2.0, 3.0], # actual color scale
           colorbar = {'title': 'Colorbar Title'})


# In[18]:


choromap2 = go.Figure(data = [data2], layout=layout)


# In[19]:


iplot(choromap2)


# In[23]:


data3 = dict(type = 'choropleth',
           locations = ['AZ', 'CA', 'NY'],
           locationmode = 'USA-states',
           colorscale = 'Greens',
           text = ['text1', 'text2', 'text3'],
           z = [1.0, 2.0, 3.0], # actual color scale
           colorbar = {'title': 'Colorbar Title'})


# In[24]:


choromap3 = go.Figure(data=[data3], layout=layout)


# In[25]:


iplot(choromap3)


# In[26]:


plot(choromap3) # open a new tap on browser


# In[27]:


data4 = dict(type = 'choropleth',
           locations = ['AZ', 'CA', 'NY'],
           locationmode = 'USA-states',
           colorscale = 'Greens',
           text = ['Arizona', 'Cali', 'New York'],
           z = [1.0, 2.0, 3.0], # actual color scale
           colorbar = {'title': 'Colorbar Title'})


# In[30]:


choromap4 = go.Figure(data=[data4], layout=layout)


# In[31]:


iplot(choromap4)


# In[32]:


import pandas as pd
df = pd.read_csv('2011_US_AGRI_Exports')


# In[33]:


df.head()


# In[40]:


data = dict(type='choropleth',
           colorscale = 'YlOrRd', # Yl: yellow, Or: orange, Rd: red
           locations = df['code'],
           locationmode = 'USA-states',
           z = df['total exports'],
           text = df['text'],
           marker = dict(line = dict(color = 'rgb(255, 255, 255)', width=2)),
           colorbar = {'title': 'Millions USD'})


# In[36]:


data


# In[41]:


layout = dict(title = '2011 US Agriculture Exports by State',
             geo = dict(scope='usa', showlakes=True, lakecolor='rgb(85,173,240)'))


# In[38]:


layout


# In[42]:


choromap5 = go.Figure(data = [data], layout=layout)


# In[43]:


iplot(choromap5)


# In[47]:


data = dict(type='choropleth',
           colorscale = 'YlOrRd', # Yl: yellow, Or: orange, Rd: red
           locations = df['code'],
           locationmode = 'USA-states',
           z = df['total exports'],
           text = df['text'],
           marker = dict(line = dict(color = 'rgb(12,12,12)', width=1)),
           colorbar = {'title': 'Millions USD'})


# In[48]:


choromap6 = go.Figure(data = [data], layout=layout)


# In[49]:


iplot(choromap6)


# In[8]:


df = pd.read_csv('2014_World_GDP')


# In[9]:


df.head()


# In[10]:


data = dict(type = 'choropleth',
           locations = df['CODE'],
           z = df['GDP (BILLIONS)'],
           text = df['COUNTRY'],
           colorbar = {'title': 'GDP in Billions USD'})


# In[20]:


layout = dict(title = '2014 Global GDP',
             geo = dict(showframe= False,
                       projection = {'type': 'natural earth'}))


# In[21]:


choromap7 = go.Figure(data=[data], layout=layout)


# In[22]:


iplot(choromap7)


# In[ ]:


'''
# reference
https://plotly.com/python/reference/#choropleth
'''

