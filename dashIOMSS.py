import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import pickle
import re
from collections import Counter
from PIL import Image
from joypy import joyplot
from streamlit_option_menu import option_menu
from dashboard_fonctions import *

st.set_page_config(layout="wide")


@st.cache
def load_data():
	continues = pickle.load(open("cont_feat.p", "rb"))
	data = pd.read_csv('viz.csv', sep='\t')
	data.drop([i for i in data if 'Unnamed' in i], axis=1, inplace=True)
	data['cash_type']=data['cash_type'].apply(lambda x:'No cash assistance received' if x=='0' else x)
	for i in continues:
		if i != 'wells':
			data[i]=data[i].astype(float)
	correl = pd.read_csv('graphs.csv')
	questions = pd.read_csv('questions.csv',sep='\t')
	questions.drop([i for i in questions.columns if 'Unnamed' in i], axis=1, inplace=True)
	quest = questions.iloc[4].to_dict()
	codes = pd.read_csv('codes.csv', index_col=None, decimal='.').dropna(how='any', subset=['color'])
	return data, correl, quest, codes

data, correl, questions, codes = load_data()

img1 = Image.open("logoAxiom.png")
img2 = Image.open("logoIOM.png")


def main():	
	#st.write(codes)
	st.sidebar.image(img1,width=200)
	st.sidebar.title("")
	st.sidebar.title("")
	
	title1, title3 = st.columns([9,2])

	with st.sidebar:
		topic = option_menu(None, ['Machine learning results', 'Correlations'],#, 'Maps application', 'Wordclouds'],
							 icons=["cpu", 'bar-chart', 'map', 'cloud'],
							 menu_icon="app-indicator", default_index=0,
							 )


	title3.image(img2)

	# ______________________________________ SHAP __________________________________#

	if topic == 'Machine learning results':
		
		title1.title('Machine learning results on predictive model trained on Questions:')
		title1.title('- How long the cash by the CFW project received lasted?')
		title1.title('- How long were the effects of the cash you received from the cash for work project?')
		st.title('')
		st.markdown("""---""")	
		st.subheader('Note:')
		st.write('A machine learning model has been run on the question related to the lasting effects of the project, '
				 'the objective of this was to identify specificaly for these question which are the parameters that'
				 ' influenced it the most. The models are run in order to try to predict as precisely as possible '
				 'the lasting effects that the respondents expressed in their response to this question. '
				 'The figures below shows which parameters have a greater impact in the prediction '
				 'of the model than a normal random aspect (following a statistic normal law)')
		st.write('')
		st.write('Each line of the graph represents one feature of the survey that is important to predict '
				 'the response to the question.')
		st.write('Each point on the right of the feature name represents one person of the survey. '
				 'A red point represent a high value to the specific feature and a blue point a '
				 'low value (a purple one a value inbetween).')
		st.write('SHAP value: When a point is on the right side, it means that it contributed to a '
				 'longer effect note while on the left side, this specific caracter of the person '
				 'reduced the final result of the prediction.')
		st.write('')
		st.write('The coding for the responses is indicated under the graph and '
				 'the interpretation of the graphs is written below.')
		st.markdown("""---""")	

		st.title('How long the cash by the CFW project received lasted?')
		temp = Image.open('shap1.png')
		image = Image.new("RGBA", temp.size, "WHITE") # Create a white rgba background
		image.paste(temp, (0, 0), temp)
		st.image(image, use_column_width = True)
		
		st.caption('Do you know how your daily salary was decided: Yes : 1 - No : 0')
		st.caption('The cash received allowed you to increase expenditures for: Meat - Responded Meat : 1 - Did not mention Meat : 0')
		st.caption("What are usually the most difficult months for your household:Febrauary - Responded February : 1 - Did not mention February : 0")

		st.write('One of the main factor for having a long effect of the cash received from the project seems to be '
				 'related to the way people used their cash:')
		st.write('- People who used it mainly for Cowpea, Kerozene, Soap, Sugar and Meat tends to have had it lasting longer')
		st.write('- On the other hand those who used it mainly on Tea leaves, Fruits and Qhat have had it lasting shorter')
		st.write('People who knew how the salary was decided also tend to have had their cash lasting longer. This coud be related to the villages'
				 'where the project was implemented as the 2 features are quite correlated as we can see on correlations.')
		st.write('When a high percentage of adults member of the household participated in the CFW activities this tends also to increase the time the cash lasted.')
		st.write('Participation of women in the decision on how to use the cash seem also to be an important factor for having the cash lasting longer')
		st.write('Finaly, it seems that February is often among the most difficult month for those who had their cash lasting longer.')

		st.markdown("""---""")

		st.title('How long were the effects of the cash you received from the cash for work project?')
		temp = Image.open('shap2.png')
		image = Image.new("RGBA", temp.size, "WHITE")  # Create a white rgba background
		image.paste(temp, (0, 0), temp)
		st.image(image, use_column_width=True)

		st.write('Here again, one of the main factor for lasting effects is the use that was made of cash:')
		st.write('- People who invested in cash, health and clothes saw longer effects')
		st.write("- People who used it mainly for Qhat had much shorter effects")
		st.write(
			'On this aspect the role of women seems to be particularly important. When women participate in the decision making on'
			'how the cash is used, the effect of the cash tends to be longer.')
		st.write('This is also the case when women participate directly to CFW activities.')
		st.write('Finaly, it seems that November is often among the most difficult month for those who have had shorter effect.')
		st.markdown("""---""")

# ______________________________________ CORRELATIONS __________________________________#

	elif topic == 'Correlations':
		sub_topic = st.sidebar.radio('Select the topic you want to look at:',['County','WASH','Shelter','Health','Other'])

		title1.title('Main correlations uncovered from the database related to '+sub_topic)

		title1.write('Note: Correlation does not mean causation. This is not because 2 features are correlated that one is '
				 'the cause of the other. So conclusion have to be made with care.')
		cat_cols = pickle.load( open( "cat_cols.p", "rb" ) )


		soustableau_correl = correl[correl['categories'].apply(lambda x: sub_topic.lower() in x)]

		st.markdown("""---""")
		k = 0
		for absc in soustableau_correl['variable_x'].unique():
			#st.write(absc)
			quest = soustableau_correl[soustableau_correl['variable_x'] == absc]
			#st.write(quest)
			for i in range(len(quest)):
				#st.write(quest.iloc[i])
				if quest.iloc[i]['filter']==quest.iloc[i]['filter']:
					if quest.iloc[i]['filter'] != 'toilet':
						df=data[data[quest.iloc[i]['filter']]=='Yes'].copy()
					else:
						df = data[data[quest.iloc[i]['filter']] != 'Bush'].copy()
				else:
					df=data.copy()
				#st.write(df.shape)
				if quest.iloc[i]['graphtype'] != 'map':
					df=select_data(quest.iloc[i],df,cat_cols)
					show_data(quest.iloc[i],df,codes)
					st.markdown("""---""")
				else:
					col1,col2,col3=st.columns([1,1,1])
					bent = px.scatter_mapbox(df[df['county']=='Bentiu'], lat="latitude", lon="longitude", color=quest.iloc[i]['variable_x'],
											color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=13.5)
					bent.update_layout(mapbox_style="open-street-map")
					bent.update_layout(margin={"r": 5, "t": 5, "l": 5, "b": 5})

					wau = px.scatter_mapbox(df[df['county'] == 'Wau'], lat="latitude", lon="longitude",
											 color=quest.iloc[i]['variable_x'],
											 color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=15)
					wau.update_layout(mapbox_style="open-street-map")
					wau.update_layout(margin={"r": 5, "t": 5, "l": 5, "b": 5})

					malak = px.scatter_mapbox(df[df['county'] == 'Malakal'], lat="latitude", lon="longitude",
											 color=quest.iloc[i]['variable_x'],
											 size_max=15, zoom=14.5)
					malak.update_layout(mapbox_style="open-street-map")
					malak.update_layout(margin={"r": 5, "t": 5, "l": 5, "b": 5})

					col1.subheader('Bentiu')
					col1.plotly_chart(bent, use_container_width=True)
					col2.subheader('Wau')
					col2.plotly_chart(wau, use_container_width=True)
					col3.subheader('Malakal')
					col3.plotly_chart(malak, use_container_width=True)

					st.write(quest.iloc[i]['Description'])

	# ______________________________________ WORDCLOUDS __________________________________#

	elif topic == 'Wordclouds':
		df = data.copy()
		text = pickle.load(open("text.p", "rb"))
		continues = pickle.load(open("cont_feat.p", "rb"))
		to_drop = pickle.load(open("drop.p", "rb"))
		quest_list = ['Reason why the selected month is difficult',
					'Do you know why you were selected to participate in this project ?',
					'Which type of training would you like to receive?',
					'Explain which changes occurred in your life thanks to the CFW',
					'Explain what happened to others youth in similar needs of CFW who did not access the program',
					'Enter livelihood category',
					'Which skills did you learn ?',
					'What are you doing NOW in terms of incomes generation?',
					'What would stop you to do more of this work ?',
					'Most important things you learnt during this cash for work project in terms of Yemeni history',
					'Most important things you learnt during this cash for work project in terms of importance of historical sites']
		child = False


		title1.title('Wordclouds for open questions')

		feature = st.selectbox(
				'Select the question for which you would like to visualize wordclouds of answers', quest_list)

# _____________________________ months difficult ______________________________________ #

		if feature == 'Reason why the selected month is difficult':
			months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
					  'November', 'December']
			feats=[i for i in df if 'reason' in i]
			col1, col2, col3 = st.columns([2, 1, 2])

			df = df[feats].applymap(lambda x : '' if x == '0' else x).copy()

			corpus=''
			for n in range(12):
				corpus += ' '.join(df[feats[n]])
				corpus = re.sub('[^A-Za-z ]', ' ', corpus)
				corpus = re.sub('\s+', ' ', corpus)
				corpus = corpus.lower()
			sw = st.multiselect('Select words you would like to remove from the wordclouds \n\n',
								[i[0] for i in Counter(corpus.split(' ')).most_common() if i[0] not in STOPWORDS][:20])

			col1, col3 = st.columns([2, 2])

			for n in range(12):
				col_corpus = ' '.join(df[feats[n]].dropna())
				col_corpus = re.sub('[^A-Za-z ]', ' ', col_corpus)
				col_corpus = re.sub('\s+', ' ', col_corpus)
				col_corpus = col_corpus.lower()
				if col_corpus == ' ' or col_corpus == '':
					col_corpus = 'No_response'
				else:
					col_corpus = ' '.join([i for i in col_corpus.split(' ') if i not in sw])
				wc = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
				wc.generate(col_corpus)
				if n%2 == 0:
					col1.subheader(months[n])
					col1.image(wc.to_array(), use_column_width=True)
				else:
					col3.subheader(months[n])
					col3.image(wc.to_array(), use_column_width=True)
# __________________________________ Learnings from Project _______________________________________________#
		elif feature in quest_list[-2:]:
			if 'Yemeni' in feature:
				colonnes=['learning1', 'learning2', 'learning3']
				titles=['First', 'Second', 'Third']
			else:
				colonnes = ['protection_learning1', 'protection_learning2', 'protection_learning3']
				titles = ['First', 'Second', 'Third']

			corpus = ' '.join(df[colonnes[0]].dropna()) + \
					 ' '.join(df[colonnes[1]].dropna()) + ' '.join(df[colonnes[2]].dropna())
			corpus = re.sub('[^A-Za-z ]', ' ', corpus)
			corpus = re.sub('\s+', ' ', corpus)
			corpus = corpus.lower()
			sw = st.multiselect('Select words you would like to remove from the wordclouds \n\n',
								[i[0] for i in Counter(corpus.split(' ')).most_common() if i[0] not in STOPWORDS][:20])
			col1, col2, col3 = st.columns([1, 1, 1])
			for i in range(3):
				col_corpus = ' '.join(df[colonnes[i]].dropna())
				col_corpus = re.sub('[^A-Za-z ]', ' ', col_corpus)
				col_corpus = re.sub('\s+', ' ', col_corpus)
				col_corpus = col_corpus.lower()
				if col_corpus == ' ' or col_corpus == '':
					col_corpus = 'No_response'
				else:
					col_corpus = ' '.join([i for i in col_corpus.split(' ') if i not in sw])
				wc = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
				wc.generate(col_corpus)
				if i == 0:
					col1.subheader(titles[0])
					col1.image(wc.to_array(), use_column_width=True)
				elif i == 1:
					col2.subheader(titles[1])
					col2.image(wc.to_array(), use_column_width=True)
				else:
					col3.subheader(titles[2])
					col3.image(wc.to_array(), use_column_width=True)
			if st.checkbox('Would you like to filter Wordcloud according to other questions'):
				feature2 = st.selectbox('Select one question to filter the wordcloud',
										[questions[i] for i in df if i not in text and i != 'UniqueID' and i not in to_drop])
				filter2 = [i for i in questions if questions[i] == feature2][0]
				if filter2 in continues:
					a = df[filter2].astype(float)
					minimum = st.slider('Select the minimum value you want to visulize',
										min_value=float(a.min()),
										max_value=float(a.max()),
										value=float(a.min())
										)
					maximum = st.slider('Select the maximum value you want to visulize', min_value=minimum,
										max_value=float(a.max()),value=float(a.max()))
					df = df[(df[filter2] >= minimum) & (df[filter2] <= maximum)]
				else:
					filter3 = st.multiselect('Select the responses you want to include',
											 [i for i in df[filter2].unique()])
					df = df[df[filter2].isin(filter3)]
				#st.write(colonnes)
				col1, col2, col3 = st.columns([1, 1, 1])
				for i in range(3):
					col_corpus = ' '.join(df[colonnes[i]].dropna())
					col_corpus = re.sub('[^A-Za-z ]', ' ', col_corpus)
					col_corpus = re.sub('\s+', ' ', col_corpus)
					col_corpus = col_corpus.lower()
					if col_corpus == ' ' or col_corpus == '':
						col_corpus = 'No_response'
					else:
						col_corpus = ' '.join([i for i in col_corpus.split(' ') if i not in sw])
					wc = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
					wc.generate(col_corpus)
					if i == 0:
						col1.subheader('Main learning')
						col1.image(wc.to_array(), use_column_width=True)
					elif i == 1:
						col2.subheader('Second main learning')
						col2.image(wc.to_array(), use_column_width=True)
					else:
						col3.subheader('Third main learning')
						col3.image(wc.to_array(), use_column_width=True)
		else:
			d = {'Do you know why you were selected to participate in this project ?' : 'why',
				'Which type of training would you like to receive?' : 'trainings',
				'Explain which changes occurred in your life thanks to the CFW' : 'changes',
				'Explain what happened to others youth in similar needs of CFW who did not access the program' : 'youth',
				'Enter livelihood category' : 'Livelihood category',
				'Which skills did you learn ?' : 'skills',
				'What are you doing NOW in terms of incomes generation?' : 'income_generation',
				'What would stop you to do more of this work ?' : 'More_work_no_explain'}
			col_corpus = ' '.join(df[d[feature]].apply(lambda x : '' if x in ['I do not know', 'There is no', 'None']
																		else x).dropna())
			col_corpus = re.sub('[^A-Za-z ]', ' ', col_corpus)
			col_corpus = re.sub('\s+', ' ', col_corpus)
			col_corpus = col_corpus.lower()
			sw = st.multiselect('Select words you would like to remove from the wordclouds \n\n',
								[i[0] for i in Counter(col_corpus.split(' ')).most_common() if i[0] not in STOPWORDS][:20])
			if col_corpus == ' ' or col_corpus == '':
				col_corpus = 'No_response'
			else:
				col_corpus = ' '.join([i for i in col_corpus.split(' ') if i not in sw])
			wc = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
			wc.generate(col_corpus)
			col1, col2, col3 = st.columns([1, 4, 1])
			col2.image(wc.to_array(), use_column_width=True)

			if st.checkbox('Would you like to filter Wordcloud according to other questions'):
				feature2 = st.selectbox('Select one question to filter the wordcloud',
										[questions[i] for i in df if
										 i not in text and i != 'UniqueID' and i not in to_drop])
				filter2 = [i for i in questions if questions[i] == feature2][0]
				if filter2 in continues:
					a=df[filter2].astype(float)
					threshold = st.slider('Select threshold value you want to visualize',
										min_value=float(a.min()),
										max_value=float(a.max()),
										value=float(a.min())
										)
					DF=[df[df[filter2] <= threshold][d[feature]], df[df[filter2] > threshold][d[feature]]]
					titres=['Response under '+str(threshold),'Response over '+str(threshold)]
				else:
					DF=[df[df[filter2] == j][d[feature]] for j in df[filter2].unique()]
					titres=['Responded : '+j for j in df[filter2].unique()]
				col1, col2 = st.columns([1, 1])
				for i in range(len(DF)):
					col_corpus = ' '.join(DF[i].dropna())
					col_corpus = re.sub('[^A-Za-z ]', ' ', col_corpus)
					col_corpus = re.sub('\s+', ' ', col_corpus)
					col_corpus = col_corpus.lower()
					if col_corpus == ' ' or col_corpus == '':
						col_corpus = 'No_response'
					else:
						col_corpus = ' '.join([i for i in col_corpus.split(' ') if i not in sw])
					wc = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
					wc.generate(col_corpus)
					if i % 2 == 0:
						col1.subheader(titres[i])
						col1.image(wc.to_array(), use_column_width=True)
					else:
						col2.subheader(titres[i])
						col2.image(wc.to_array(), use_column_width=True)

if __name__ == '__main__':
	main()
