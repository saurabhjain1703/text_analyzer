
# coding: utf-8

# In[1]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pyttsx3
import string
from nltk.corpus import stopwords
import re
from nltk.sentiment.util import *
def speech(a):
	print(a)
	engine=pyttsx3.init()
	engine.setProperty('rate',120)
	engine.setProperty('volume',1.0)
	engine.say(a)
	engine.runAndWait()
	return ""

text=input(speech("Enter the Text you wanna analyze"))
speech("you have entered "+text)

#remove whitespaces
# def remove_whitespace(text):
#     return " ".join(text).split()
# b = remove_whitespace(text)

# remove numbers
def remove_numbers(text):
    return re.sub(r'\d+', '',str(text))
a = remove_numbers(text)
#print(d)

# remove url
def remove_url(a):
    #b.replace('""','')
    return re.sub(r"http\S+", "", a)
c = remove_url(a)

# # remove punctuation
def remove_punctuation(c):
    words = nltk.word_tokenize(c)
    punt_remove = [w for w in words if w.lower() not in string.punctuation]
    return " ".join(punt_remove)
d = remove_punctuation(c)

def remove_stopwords(d,lang='english'):
    words=nltk.word_tokenize(d)
    lang_stopwords=stopwords.words(lang)
    stopwords_removed=[w for w in words if w.lower() not in lang_stopwords]
    return " ".join(stopwords_removed)
f=remove_stopwords(d)

print(f)

S=SentimentIntensityAnalyzer()
x=S.polarity_scores(f)['compound']
p=S.polarity_scores(f)['pos']
n=S.polarity_scores(f)['neg']
print(x,p,n)
if x<0:
	speech('your Text is Conveying negative sentiment')
	t="Negativity score is "+str(round(n*100))+"percent"
	speech(t)
elif x>0:
	speech('your Text is Conveying Positive sentiment')
	t="Positivity score is "+str(round(p*100))+"percent"
	speech(t)
else:
	speech('your Text is Conveying neutral sentiment')
	speech("Overall score is  Zero")
		

