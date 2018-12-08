
# coding: utf-8

# In[ ]:


import pandas as pd
from pandas import DataFrame
from nltk import SnowballStemmer
import tweepy
import nltk
import re
from sklearn.cross_validation import train_test_split
import string
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
import matplotlib.pyplot as plt
import pyttsx3
def speech(a):
	print(a)
	engine=pyttsx3.init()
	engine.setProperty('rate',120)
	engine.setProperty('volume',1.0)
	engine.say(a)
	engine.runAndWait()
	return ""
intro="Welcome to Tweet Analyser Project Made By Mohit Singh,SAurabh jain,mukul dubey and maneesh"
conn="Please wait while establishing connection"
fetch="fetching the tweet"
analys="Analysing Tweets for Sentiment Analysis "
pt="plotting the stats  "
speech(intro)
# provide your access details below
access_token="956033448577241089-6S1DYfuZdl55BtGnArUHw8ofVWS4Cep"
access_token_secret="2gwgxEJC9OxsdhSgVRjcprzZY3GxFccb42LvSDSGAJkIK"
consumer_key="flGGNLwGmqNiav5ooMwgkIklF"
consumer_secret="FzSANGquLnPSlqAZAutGy7hbjEq0S3FrVFwzkrGEDKsp5OLirW"
# establish  a connection
speech(conn)
auth=tweepy.auth.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)
api=tweepy.API(auth)

# fetch recent tweets
speech(fetch)
fetched_tweets=api.search(q=[input(speech("Enter the name to Search for : ")),input(speech("Field You want to Search for"))],result_type='recent',lang='en',count=input(speech("Enter Number of tweets to analyse and Wait While Fetching")))
print("Number of tweets:",len(fetched_tweets))	

# df1=df.values
# for i in range(len(fetched_tweets)):
#     b=df1[i][2]
    
#     #remove whitespaces
#     def remove_whitespace(text):
#         return " ".join(string(text).split())
#     a = remove_whitespace(text)

    # remove numbers
#     def remove_numbers(text):
#         return re.sub(r'\d+', '',str(text))
#     b = remove_numbers(text)

    # remove url
   

 #print the tweet text
tweets=[]
a=[]
speech("showing Tweets Respective to their Tweet Id:")
for c in fetched_tweets:
    a=c.text
    
    def remove_numbers(a):
        return re.sub(r'\d+', '',str(a))
    c = remove_numbers(a)
      #print(c.text)
   
    def remove_url(c):
        #b.replace('""','')
        return re.sub(r"http\S+", "", str(c))
    d = remove_url(c)
    #print(d)

    #remove punctuation
    def remove_punctuation(d):
        words = nltk.word_tokenize(d)
        punt_remove = [w for w in words if w.lower() not in string.punctuation]
        return " ".join(punt_remove)
    e=remove_punctuation(d)
    
    def remove_stopwords(e,lang='english'):
        words=nltk.word_tokenize(e)
        lang_stopwords=stopwords.words(lang)
        stopwords_removed=[w for w in words if w.lower() not in lang_stopwords]
        return " ".join(stopwords_removed)
    f=remove_stopwords(e)
    
    if f not in tweets:
        tweets.append(f)
        
    
#print(tweets)
df=pd.DataFrame({'text':tweets})
print(df.text)
	# print('Tweet ID:',b.id,'Tweets:',d,'\n')
	#print('Tweet Text:',tweet.text)
	#print()

# Save features to Dataframe
speech("Copying fetched tweets to Data frame")
def populate_tweet_df(tweets):
	# create an empty dataframe
	df=pd.DataFrame()
	df['id']=list(map(lambda tweet:tweet.id,tweets))
	df['text']=list(map(lambda tweet:tweet.text,tweets))
	df['retweeted']=list(map(lambda tweet:tweet.retweeted,tweets))
	df['place']=list(map(lambda tweet:tweet.user.location,tweets))
	df['screen_name']=list(map(lambda tweet:tweet.user.screen_name,tweets))
	df['verified_user']=list(map(lambda tweet:tweet.user.verified,tweets))
	df['followers_count']=list(map(lambda tweet:tweet.user.followers_count,tweets))
	df['friends_count']=list(map(lambda tweet:tweet.user.friends_count,tweets))
	#df['friendship_coeff']=list(map(lambda tweet:float(tweet.user.followers_count)%float(tweet.user.friends_count++++++),tweets))
	return df
df=populate_tweet_df(fetched_tweets)
print()


df.to_csv("E:/twitter.csv")
speech(analys)
SIA=SentimentIntensityAnalyzer()
df['polarity']=df.text.apply(lambda x:SIA.polarity_scores(x)['compound'])
df['nuetral_score']=df.text.apply(lambda x:SIA.polarity_scores(x)['neu'])
df['negative_score']=df.text.apply(lambda x:SIA.polarity_scores(x)['neg'])
df['positive_score']=df.text.apply(lambda x:SIA.polarity_scores(x)['pos'])
df['sentiment']=''
df.loc[df.polarity>0,'sentiment']='POSITIVE'
df.loc[df.polarity==0,'sentiment']='NEUTRAL'
df.loc[df.polarity<0,'sentiment']='NEGATIVE'
print(df.head())



"""import sklearn.preprocessing as pp
lb=pp.LabelBinarizer()
df1=pd.get_dummies(df.sentiment,prefix="sentiment")

df=pd.concat([df,df1],axis=1)

X=df.sentiment_POSITIVE
Y=df.sentiment_NEGATIVE


#split data into Training and Testing
# Training Set= 80% of actual data
from sklearn.cross_validation import train_test_split
# Testing Set= 20% of actual data
X_train,X_test,Y_train,Y_test=train_test_split(x,y,train_size=0.8)
print("X Trainig:",X_train.shape)
print("X Testing:",X_test.shape)
print("Y Trainig:",Y_train.shape)
print("Y Trainig:",Y_test.shape)


X_train=X_train.reshape(20,2)
Y_train=Y_train.reshape(20,2)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
clf=KNeighborsClassifier(n_neighbors=5,p=2,metric='minkowski')
clf.fit(Y_train,X_train)
print("--------------------------------------------------------------")
print("Accuracy:")
print(1+(r2_score(X_train,clf.predict(Y_train))))"""

chart=input(speech('You want to plot pie  or bar graph'))
speech("plotting "+chart+"graph as stats")
df.sentiment.value_counts().plot(kind=chart,title="Sentiment Analysis")
plt.show()




