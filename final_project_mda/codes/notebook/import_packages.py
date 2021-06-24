#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
# Make warnings quite
pd.options.mode.chained_assignment = None 
import re

import matplotlib.pyplot as plt
import geopandas

#Libraries used in .py file
#!pip install boto3
import boto3
import os



from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import plotly.graph_objects as go 



import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
import spacy
import spacy.cli
spacy.cli.download("en_core_web_sm")
import string

#!pip install pyldavis
#!pip install --upgrade pandas==1.2


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.sklearn

import plotly.graph_objects as go
import plotly.express as px

from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer


nltk.download('vader_lexicon')

#!pip install geopandas
#!pip install -U plotly

import geopandas as gpd

import plotly
import plotly.express as px


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


from scipy.cluster.hierarchy import linkage,fcluster
from scipy.cluster.vq import kmeans, vq
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.vq import whiten
import seaborn as sns

