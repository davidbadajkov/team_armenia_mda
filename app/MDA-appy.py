import streamlit as st
from streamlit import components  

import pandas as pd 
import numpy as np 

from pprint import pprint

import matplotlib.pyplot as plt 

import seaborn as sns
sns.set_style('darkgrid')

from plotly.subplots import make_subplots
from plotly.offline import plot 
import plotly.graph_objects as go 
import plotly.express as px 

#import boto3
import s3fs
import os

# before deploying the app I have to get the data from a remote server, it can't be stored in my local machine
# importing transcripts. NOTE: this one is not definitive version
#df = pd.read_csv('https://s3grouparmenia.s3.eu-central-1.amazonaws.com/1405transcript_preprocessed_compact.csv')
df = pd.read_csv('https://s3grouparmenia.s3.eu-central-1.amazonaws.com/Transcripts_with_sentiment_topic')
df.dropna(axis = 0, how = 'any', inplace = True)
df['Year_x'] = [int(i) for i in df['Year_x']]

fs = s3fs.S3FileSystem(anon = False)

@st.cache(ttl=600)
def read_file(filename):
	with fs.open(filename) as f:
		return f.read().decode('utf-8')

## setting up page width
st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: 1200px;
        
    }}
    
</style>
""",
        unsafe_allow_html=True,
    )
## main header
st.title('Analysis of UN General Assemblies 1970-2018')

st.markdown(""" 
	The General Debate in the General Assembly is the most elaborate and important General Debate, where every delegation's top
	senior official delivers a statement. 
	Although it is called a Debate, it is not the place to discuss and respond to other delegations' statements. It is an opportunity 
	for the nations to raise ideas for discussion, make announcements and propose motions.
	\n
	In this application we dive deep into the contents of the United Nations General Debate speech transcripts for the past 50 years.
	We aim to identify the general themes that have been addressed in the debates, see how the main discussion themes have evolved
	throughout the years and contextualize the trend with external data for select topics we considered relevant. 
	""")

# descriptive section

# setting up sidebar

st.sidebar.header('Filter the UN speeches')
unique_years = sorted(list(df['Year_x'].unique()), reverse = True)
unique_countries = sorted(list(df['Country_x'].unique()))
unique_sessions = sorted(list(df['Session_x'].unique()))
selected_years = st.sidebar.slider('Year', 1970, 2018, (1970, 2018))
country_expander = st.sidebar.beta_expander('Filter countries')
with country_expander:
	selected_country = st.multiselect('Country', unique_countries, default = unique_countries)
session_expander = st.sidebar.beta_expander('Filter sessions')
with session_expander:
	selected_session = st.multiselect('Session', unique_sessions, default = unique_sessions)

filtered_df = df[(df.Country_x.isin(selected_country)) & (df.Session_x.isin(selected_session)) & (df.Year_x.between(selected_years[0], selected_years[1]))]

st.write("""
	## Visualizing major themes
	Below we visualize the top 10 most common bi-grams in the dataset. These bi-grams shed some light as to what
	at are the most common themes addressed since 1970.
	""")

html_file2 = read_file('s3grouparmenia/bigrams.html')
components.v1.html(html_file2, width = 1200, height = 450)

st.write("""
	### LDA topic modelling
	Here we run the LDA algorithm to identify the top main topics in the dataset by clustering the words in the document.
	We assign a dominant topic to each filing, and name the topic manually. \n
	**Note:** The model shown below is pre-computed and hence the sidebar filters will not work. We freeze the LDA output
	in order to perform a trend analysis on the resulting topics. Otherwise, depending on the specified filters, the model may 
	output different topics which would then need to me named. 
	""")


html_file = read_file('s3grouparmenia/lda.html')
components.v1.html(html_file, width = 1200, height = 800, scrolling = True)

topic_nums = [1,2,3,4,5,6,7,8]
topic_names = ['Development of Africa', 'Human Rights', 'International Security', 'Nuclear Politics',
				'Economic Development', 'Israel-Palestine Conflict', 'World Peace', 'Sustainable Development' ]
topic_df = pd.DataFrame(
	{'Topic Number': topic_nums,
	'Topic Names': topic_names})

# setting up data for plotting topic trends
res = {}
names = filtered_df['topic_name'].unique()
topic_dic = {key:None for key in names}
years = list(np.sort(filtered_df['Year_x'].unique()))
for i in list(topic_dic.keys()):
	topic_dic[i] = years

for topic in topic_dic.items():
	res[topic[0]] = {}
	for year in topic[1]:
		res[topic[0]][year] = len(filtered_df.loc[(filtered_df.topic_name == topic[0]) & (filtered_df.Year_x == year)])
df2 = pd.DataFrame(res)

st.write("""
	### Visualizing topic trends
	Use the sidebar to the left to filter the transcripts.
	""")
topics = st.multiselect('Choose a topic name', topic_names, default = ['Sustainable Development', 'Israel-Palestine Conflict'])
topics_lc = [x.lower() for x in topics]
pd.options.plotting.backend = 'plotly'
filt_df = df2[topics_lc]
filt_df.index.name = 'Years'
fig = filt_df.plot()

# this is a hack to center the table
col1, col2 = st.beta_columns((1,1))
with col1:
	st.write("""
 		#### Topic names
 		We manually label the topics based on the top 30 salient words shown above. The topic names are shown in the table below.  
		""")
	st.dataframe(topic_df)
with col2:
	st.write("""
		#### Topic trends """)
	st.write(fig)

st.write("""
	In the following sections we dig deeper into two of the topics that we identified.
	""")

st.write("""
	### Sustainable Development
	In this section we aim to measure how addressing sustainable development during the General Assembly is reflected 
	in practice and policy.
	""")

map_df = pd.read_csv('https://s3grouparmenia.s3.eu-central-1.amazonaws.com/map_df.csv')
worst_performers = map_df.sort_values(by = ['index'], ascending = False).head(10)
top_performers = map_df.loc[map_df['count'] > 1].sort_values(by = ['index']).dropna().head()
performance = pd.concat([worst_performers, top_performers])

@st.cache
def plot_map(df, var):
	fig = px.choropleth(df,
		geojson = df.geometry,
		locationmode = 'ISO-3',
		locations = df.iso_a3,
		color = var,
		color_continuous_scale = 'blues',
		projection = 'orthographic',
		hover_name = 'name',
		hover_data = ['SDI_mean', 'index', 'count'])
	fig.update_geos(fitbounds = 'locations', visible = True)
	return fig

col1, col2 = st.beta_columns((1,1))
with col1:
	labs = ['Sustainable Development Index (SDI)', "Honesty Ratio by Country", 'Speech Count', 'Best/Worst Performers']
	rad = st.radio('Choose the variable to show in the map', labs, index =0)
	if rad == 'Sustainable Development Index (SDI)':
		st.markdown("""
			See [here](https://www.sustainabledevelopmentindex.org/methods) for more detail on how the sustainable development index is calculated.
			""")
	elif rad == "Honesty Ratio by Country":
		st.markdown("""
			This index measures how well the countries back up their words with actions and policies in the real world.
			We measure this with the following formula:\n
			""")
		latext = r'''
		$$
		Honesty\: Ratio = \frac{speech\: count_{scaled}}{mean\: SDI}
		$$
		'''
		st.write(latext)
		st.write("where")

		latext2 = r'''
		$$
		speech\: count_{scaled} = \frac{speech\: count - speech\: count_{min}}{speech\:count_{max} - speech\:count_{min}}
		$$
		'''
		st.write(latext2)
		st.write("""
			Smaller values of the index are evidence that the claims of the countries are substantiated by programs
			to create a sustainable environment and tackle climate change problems. 
			""")
	elif rad == 'Speech Count':
		st.write("""
			The map shows the number of times each country has given a speech centered around sustainable development
			between 1990 and 2018 to match the measurement interval of the sustainable development index.\n
			""")
	elif rad == 'Best/Worst Performers':
		tab = performance[['name', 'count', 'SDI_mean', 'index']]
		tab.rename(columns = {'name': 'Country', 'count': 'Speech Count', 'SDI_mean': 'Mean SDI', 'index': 'Honesty Ratio'}, inplace = True)
		tab.reset_index(inplace = True)
		tab.drop(columns = 'index', inplace = True)
		st.dataframe(tab)

with col2:
	if rad == 'Sustainable Development Index (SDI)':
		fig = plot_map(map_df, 'SDI_mean')
		st.plotly_chart(fig)
	elif rad == "Honesty Ratio by Country":
		fig = plot_map(map_df, 'index')
		st.plotly_chart(fig)
	elif rad == 'Speech Count':
		fig = plot_map(map_df, 'count')
		st.plotly_chart(fig)
	elif rad == 'Best/Worst Performers':
		fig = plot_map(performance, 'index')
		fig = px.choropleth(performance,
			geojson = performance.geometry,
			locationmode = 'ISO-3',
			locations = performance.iso_a3,
			color = 'index',
			color_continuous_scale = 'reds',
			projection = 'orthographic',
			hover_name = 'name',
			hover_data = ['SDI_mean', 'count', 'index'])
		fig.update_geos(fitbounds = 'locations', visible = True)
		st.plotly_chart(fig)

st.write("""
	### Israel-Palestine Conflict
	Here we cluster the countries according to the underlying sentiment in their speeches that address the Israel
	and Palestine conflicts. We evaluate the transcripts per decade since 1970 in order to observe how their stance changes
	throughout the years. \n
	We measure the sentiment by using the following measures: \n
	* Polarity: shows the general sentiment in the speech. It is computed from the positive, negative and neutral scores. \n
	* Subjectivity: measures the degree to which subjective emotions are expressed in the speech. \n
	* Positivity: a score determined based on the degree to which positive words are used. \n
	* Neutrality: a score based on words that do not fall under any of the positive or negative categories. \n
	* Negativity: similar to positivity, but for negative words.
	""")

cluster_data = pd.read_csv('https://s3grouparmenia.s3.eu-central-1.amazonaws.com/clusters_with_geo_data+(1)')
cluster_data.rename(columns = {'cluster_labels_1': 'Clusters 1970-1980', 
	'cluster_labels_2': 'Clusters 1980-1990',
	'cluster_labels_3': 'Clusters 1990-2000',
	'cluster_labels_4': 'Clusters 2000-2010',
	'cluster_labels_5': 'Clusters 2010-present'}, inplace = True)

@st.cache
def map_clusters(df, period):
	fig = px.choropleth(df,
		geojson = df.geometry,
		locationmode = 'ISO-3',
		locations = df.country,
		color = period,
		color_discrete_map = {'0.0': 'red', '1.0': 'yellow', '2.0': 'green'},
		projection = 'orthographic',
		hover_name = 'name')
	fig.update_geos(fitbounds = 'locations', visible = True)
	return fig 

col1, col2 = st.beta_columns((1,1))
variables = ['scaled_polarity', 'scaled_subjectivity', 'scaled_positivity', 'scaled_negativity', 'scaled_neutrality']

with col1:
	labs1 = ['1970-1980', '1980-1990', '1990-2000', '2000-2010', '2010-present']
	but = st.radio('Choose timeframe:', labs1, index = 0)

	if but == '1970-1980':
		df = pd.read_csv('https://s3grouparmenia.s3.eu-central-1.amazonaws.com/clusters_per_decade/1970-1980.csv')
		df.rename(columns = {'cluster_labels_1': 'Clusters', 'scaled_pos': 'scaled_positivity', 'scaled_neg': 'scaled_negativity', 'scaled_neu': 'scaled_neutrality'}, inplace = True)
		grpd = df.groupby('Clusters')[variables].mean()
		fig = px.bar(grpd, x = grpd.index, y = ['scaled_polarity', 'scaled_subjectivity', 'scaled_positivity', 'scaled_negativity', 'scaled_neutrality'], barmode = 'group', width = 600)
		st.plotly_chart(fig, width = 600)

	if but == '1980-1990':
		df = pd.read_csv('https://s3grouparmenia.s3.eu-central-1.amazonaws.com/clusters_per_decade/1980-1990.csv')
		df.rename(columns = {'cluster_labels_2': 'Clusters', 'scaled_pos': 'scaled_positivity', 'scaled_neg': 'scaled_negativity', 'scaled_neu': 'scaled_neutrality'}, inplace = True)
		grpd = df.groupby('Clusters')[variables].mean()
		fig = px.bar(grpd, x = grpd.index, y = ['scaled_polarity', 'scaled_subjectivity', 'scaled_positivity', 'scaled_negativity', 'scaled_neutrality'], barmode = 'group', width = 600)
		st.plotly_chart(fig, width = 600)

	if but == '1990-2000':
		df = pd.read_csv('https://s3grouparmenia.s3.eu-central-1.amazonaws.com/clusters_per_decade/1990-2000.csv')
		df.rename(columns = {'cluster_labels_3': 'Clusters', 'scaled_pos': 'scaled_positivity', 'scaled_neg': 'scaled_negativity', 'scaled_neu': 'scaled_neutrality'}, inplace = True)
		grpd = df.groupby('Clusters')[variables].mean()
		fig = px.bar(grpd, x = grpd.index, y = ['scaled_polarity', 'scaled_subjectivity', 'scaled_positivity', 'scaled_negativity', 'scaled_neutrality'], barmode = 'group', width = 600)
		st.plotly_chart(fig, width = 600)

	if but == '2000-2010':
		df = pd.read_csv('https://s3grouparmenia.s3.eu-central-1.amazonaws.com/clusters_per_decade/2000-2010.csv')
		df.rename(columns = {'cluster_labels_4': 'Clusters', 'scaled_pos': 'scaled_positivity', 'scaled_neg': 'scaled_negativity', 'scaled_neu': 'scaled_neutrality'}, inplace = True)
		grpd = df.groupby('Clusters')[variables].mean()
		fig = px.bar(grpd, x = grpd.index, y = ['scaled_polarity', 'scaled_subjectivity', 'scaled_positivity', 'scaled_negativity', 'scaled_neutrality'], barmode = 'group', width = 600)
		st.plotly_chart(fig, width = 600)

	if but == '2010-present':
		df = pd.read_csv('https://s3grouparmenia.s3.eu-central-1.amazonaws.com/clusters_per_decade/2010-present.csv')
		df.rename(columns = {'cluster_labels_5': 'Clusters', 'scaled_pos': 'scaled_positivity', 'scaled_neg': 'scaled_negativity', 'scaled_neu': 'scaled_neutrality'}, inplace = True)
		grpd = df.groupby('Clusters')[variables].mean()
		fig = px.bar(grpd, x = grpd.index, y = ['scaled_polarity', 'scaled_subjectivity', 'scaled_positivity', 'scaled_negativity', 'scaled_neutrality'], barmode = 'group', width = 600)
		st.plotly_chart(fig, width = 600)		

with col2:
	if but == '1970-1980':
		df3 = cluster_data[['name', 'geometry', 'country', 'Clusters 1970-1980']]
		df3.dropna(axis = 0, how = 'any', inplace = True)
		df3['Clusters 1970-1980'] = df3['Clusters 1970-1980'].astype('Int64')
		df3['Clusters 1970-1980'] = df3['Clusters 1970-1980'].apply(str)
		fig = map_clusters(df3, 'Clusters 1970-1980')
		st.plotly_chart(fig)

	elif but == '1980-1990':
		df3 = cluster_data[['name', 'geometry', 'country', 'Clusters 1980-1990']]
		df3.dropna(axis = 0, how = 'any', inplace = True)
		df3['Clusters 1980-1990'] = df3['Clusters 1980-1990'].astype('Int64')
		df3['Clusters 1980-1990'] = df3['Clusters 1980-1990'].apply(str)
		fig = map_clusters(df3, 'Clusters 1980-1990')
		st.plotly_chart(fig)

	elif but == '1990-2000':
		df3 = cluster_data[['name', 'geometry', 'country', 'Clusters 1990-2000']]
		df3.dropna(axis = 0, how = 'any', inplace = True)
		df3['Clusters 1990-2000'] = df3['Clusters 1990-2000'].astype('Int64')
		df3['Clusters 1990-2000'] = df3['Clusters 1990-2000'].apply(str)
		fig = map_clusters(df3, 'Clusters 1990-2000')
		st.plotly_chart(fig)

	elif but == '2000-2010':
		df3 = cluster_data[['name', 'geometry', 'country', 'Clusters 2000-2010']]
		df3.dropna(axis = 0, how = 'any', inplace = True)
		df3['Clusters 2000-2010'] = df3['Clusters 2000-2010'].astype('Int64')
		df3['Clusters 2000-2010'] = df3['Clusters 2000-2010'].apply(str)
		fig = map_clusters(df3, 'Clusters 2000-2010')
		st.plotly_chart(fig)

	elif but == '2010-present':
		df3 = cluster_data[['name', 'geometry', 'country', 'Clusters 2010-present']]
		df3.dropna(axis = 0, how = 'any', inplace = True)
		df3['Clusters 2010-present'] = df3['Clusters 2010-present'].astype('Int64')
		df3['Clusters 2010-present'] = df3['Clusters 2010-present'].apply(str)
		fig = map_clusters(df3, 'Clusters 2010-present')
		st.plotly_chart(fig)
	


