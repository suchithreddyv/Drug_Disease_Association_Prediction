#!/usr/bin/env python
# coding: utf-8

import numpy as nmp
import pandas as pnd
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as mplt
import seaborn as sb
from sklearn import metrics


# reading the data
train_data = pnd.read_csv('drugsComTrain_raw.csv')
test_data = pnd.read_csv('drugsComTest_raw.csv')

train_data.head()

print("Shape (rows and columns) of training set :", train_data.shape)

test_data.head()

print("Shape (rows and columns) of testing set :", test_data.shape)


train_data.info()


train1_data=train_data[['drugName', 'condition' , 'review', 'rating','date','usefulCount']] 


train1_data.head()

test1_data=test_data[['drugName', 'condition' , 'review', 'rating','date','usefulCount']] 

test1_data.head()

dframe = pnd.concat([train1_data, test1_data])

dframe.head()

dframe.info()

dframe.shape

dframe.isnull().any()

# checking the most popular drugs per conditions

mydf=dframe.groupby(['condition'])['drugName'].nunique().sort_values(ascending = False).head(40)
ax=mydf.plot.bar(figsize = (19, 7), color = 'blue')
mplt.title('Most drugs available per Conditions in the Patients', fontsize = 30)
mplt.xlabel('Conditions', fontsize = 20)
mplt.ylabel('count')
for p in ax.patches:
    ax.annotate(str(p.get_height()),(p.get_x()*1.002,p.get_height()*1.020))
mplt.show()

# checking the different types of conditions patients


mydf2=dframe['condition'].value_counts().head(40)
ax2=mydf2.plot.bar(figsize = (19, 7), color = 'purple')
mplt.title('Most Common Conditions in the Patients', fontsize = 30)
mplt.xlabel('Conditions', fontsize = 20)
mplt.ylabel('count')
for p in ax2.patches:
    ax2.annotate(str(p.get_height()),(p.get_x()*1.002,p.get_height()*1.020),rotation=45)
mplt.show()

# let's read some reviews

train_data['review'][5]
dframe['rating'].value_counts()
size = [68005,36708, 25046, 12547, 8462, 10723, 6671, 8718, 9265, 28918]
colors = ['pink', 'black',  'purple', 'orange', 'skyblue', 'green', 'yellow','blue','red', 'cyan']
labels = "10", "9", "8", "7", "6", "5", "4", "3", "2", "1"

my_pie = mplt.Circle((0, 0), 0.7, color = 'white')

mplt.rcParams['figure.figsize'] = (10, 10)
mplt.pie(size, colors = colors, labels = labels, autopct = '%.2f%%')
mplt.axis('off')
mplt.title('A Pie Chart Representing the Share of Ratings', fontsize = 30)
p = mplt.gcf()
mplt.gca().add_artist(my_pie)
mplt.legend()
mplt.show()
dframe.loc[(dframe['rating'] >= 5), 'Sentiment_review'] = 1
dframe.loc[(dframe['rating'] < 5), 'Sentiment_review'] = 0
dframe['Sentiment_review'].value_counts()

dframe.head()
# a pie chart to represent the sentiments of the patients

size = [161491, 53572]
colors = ['orange', 'skyblue']
labels = "Positive Sentiment","Negative Sentiment"
explode = [0, 0.1]

mplt.rcParams['figure.figsize'] = (10, 10)
mplt.pie(size, colors = colors, labels = labels, explode = explode, autopct = '%.2f%%')
mplt.axis('off')
mplt.title('A Pie Chart Representing the Sentiments of Patients', fontsize = 30)
mplt.legend()
mplt.show()
dframe=dframe.drop(['date'],axis=1)
dframe.head()
dframe['condition'].isnull().sum()

dframe = dframe.dropna(axis = 0)
dframe = dframe.drop(['usefulCount'],axis=1)
dframe.shape
dframe.head()

import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer

mystopwords = set(stopwords.words('english'))

non_stopwords = ["aren't","couldn't","didn't","doesn't","don't","hadn't","hasn't","haven't","isn't","mightn't",
            "mustn't","needn't","no","nor","not","shan't","shouldn't","wasn't","weren't","wouldn't"]
for i in non_stopwords:
    mystopwords.remove(i)

dframe['review'].head()

dframe_condition = dframe.groupby(['condition'])['drugName'].nunique().sort_values(ascending=False)
dframe_condition = pnd.DataFrame(dframe_condition).reset_index()
dframe_condition

# setting a df with conditions with only one drug
dframe_condition_1 = dframe_condition[dframe_condition['drugName'] == 1].reset_index()
all_list = set(dframe.index)

# deleting them
condition_list = []
for i,j in enumerate(dframe['condition']):
    for c in list(dframe_condition_1['condition']):
        if j == c:
            condition_list.append(i)
new_idx = all_list.difference(set(condition_list))
dframe = dframe.iloc[list(new_idx)].reset_index()
del dframe['index']

all_list = set(dframe.index)
span_list = []
for i,j in enumerate(dframe['condition']):
    if '</span>' in j:
        span_list.append(i)
        
new_idx = all_list.difference(set(span_list))
dframe = dframe.iloc[list(new_idx)].reset_index()
del dframe['index']

stemmer = SnowballStemmer('english')

def review_to_words(raw_review):
    # 1. Delete HTML 
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    # 2. Make a space
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    # 3. lower letters
    words = letters_only.lower().split()
    # 5. Stopwords 
    meaningful_words = [w for w in words if not w in mystopwords]
    # 6. Stemming
    stemming_words = [stemmer.stem(w) for w in meaningful_words]
    # 7. space join words
    return( ' '.join(stemming_words))

dframe['review_clean'] = dframe['review'].apply(review_to_words)
dframe.head()

dframe= dframe.drop(['review'],axis=1)

dframe.head()

x=dframe.iloc[:,dframe.columns!='Sentiment_review']
y=dframe.iloc[:,dframe.columns=='Sentiment_review'].values.ravel()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix,classification_report

cvz = CountVectorizer(max_features = 20000, ngram_range = (5, 5))
pipeline = Pipeline([('vect',cvz)])

dframe_train_features = pipeline.fit_transform(x['review_clean'],x['drugName'])
#print(cvz.vocabulary_)


from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dframe_train_features,y, test_size = 0.3, random_state=0)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

predictlog=logreg.predict(X_test)
logreg.coef_

from sklearn import metrics

model1=metrics.accuracy_score(y_test,predictlog)

print(model1)

cnfmat1=confusion_matrix(y_test,predictlog)

labels = [0,1]
sb.heatmap(cnfmat1, annot=True, cmap="YlGnBu", fmt=".05f", xticklabels=labels, yticklabels=labels)
mplt.show()

cr1=classification_report(y_test,predictlog)
print(cr1)

from sklearn.svm import SVC

svm=SVC()

svm.fit(X_train, y_train)
predictsvm=svm.predict(X_test)
model2=metrics.accuracy_score(y_test,predictsvm)
print(model2)

cnfmat2=confusion_matrix(y_test,predictsvm)
labels = [0,1]
sb.heatmap(cnfmat2, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
mplt.show()
cr2=classification_report(y_test,predictsvm)
print(cr2)

from sklearn.neighbors import KNeighborsClassifier
k_neighbour = KNeighborsClassifier(n_neighbors=2)
k_neighbour.fit(X_train, y_train)

predictknn=k_neighbour.predict(X_test)

model3=metrics.accuracy_score(y_test,predictknn)
print(model3)

cnfmat3=confusion_matrix(y_test,predictknn)

labels = [0,1]
sb.heatmap(cnfmat3, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
mplt.show()

cr3=classification_report(y_test,predictknn)
print(cr3)

import matplotlib.pyplot as mplt; mplt.rcdefaults()

myobjects = (' LogisticRegression','SVM','KNeighbors')
y_pos = nmp.arange(len(myobjects))
performance = [model1,model2,model3]

ax3=mplt.bar(y_pos, performance, align='center', alpha=0.5)
mplt.xticks(y_pos, myobjects)
mplt.ylabel('Accuracy')
mplt.title('LogisticRegression vs SVM vs KNeighborsClassifier' )
for p in ax3.patches:
    mplt.annotate(str(round(p.get_height(),5)),(p.get_x()*1.015,p.get_height()*1.010))
mplt.show()

#test1_data['testreview_clean']=test1_data['review'].apply(review_to_words)
#test1_data.head()

#test1_data_features = pipeline.fit_transform(test1_data['testreview_clean'],test1_data['drugName'])

#testpredict=svm.predict(test1_data_features)

#print(testpredict)
#testres=pnd.DataFrame(testpredict,columns=['predict'])

#testres['predict'].value_counts()

