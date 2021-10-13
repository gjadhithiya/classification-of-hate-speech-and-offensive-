from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.metrics import classification_report
import pandas, numpy, string
import pickle as pkl
import pandas as pd
import nltk
nltk.download('punkt')
import regex as re
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    
    return emoji_pattern.sub(r'', text)

def clean_text(text ): 
    delete_dict = {sp_character: '' for sp_character in string.punctuation} 
    delete_dict[' '] = ' ' 
    table = str.maketrans(delete_dict)
    text1 = text.translate(table)
    textArr= text1.split()
    text2 = ' '.join([w for w in textArr if ( not w.isdigit() and  ( not w.isdigit() and len(w)>3))]) 
    
    return text2.lower()
    
from nltk import sent_tokenize
import json
import random
texts = pd.read_csv('/content/sample_data/tamil_offensive_train.csv',)
print(texts.shape)
def tokenize(text):
    tokenized = sent_tokenize(text)
    return tokenized
def shuffle_tokenized(text):
    random.shuffle(text)
    newl = list(text)
    shuffled.append(newl)
    return text
augmented = []
reps = []
spam_reviews = texts.loc[texts['category'] == 'OFF']
print("spam_reviews",spam_reviews)
for ng_rev in spam_reviews['text']:
    emr_text=remove_emoji(ng_rev)
    cln_text=clean_text(emr_text)
   # print("ng_rev: ",ng_rev)
    tok = tokenize(cln_text)
    shuffled = [tok]
    # print(ng_rev)
    for i in range(2):
        # generate 10 new reviews
        shuffle_tokenized(shuffled[-1])
    for k in shuffled:
      s = ' '
      new_rev = s.join(k)
      if new_rev not in augmented:
        augmented.append([new_rev, "OFF"])
      else:
        reps.append(new_rev)
more_spam = pd.DataFrame(augmented, columns=['text', 'category'])
# print(more_spam.head())

texts = pd.concat([texts, more_spam])
texts = texts.sample(frac=1).reset_index(drop=True)
print(texts.head())
print(texts.shape)
test_text=pd.read_excel('/content/tam_offesive_test.xlsx')
#print(texts)
print(test_text)
counts = texts['category'].value_counts()
print(counts)
res = texts[~texts['category'].isin(counts[counts < 5].index)]
#print(texts)
print(res)
counts = res['category'].value_counts()
print(counts)
train_tweet=res['text']
print(train_tweet)
print(train_tweet.shape)
train_label=res['category']
print(train_label)
print(train_label.shape)
print(type(train_tweet))
train_x=train_tweet[:]
train_y=train_label[:]
print(" train_x :",train_x )
print(" train_y :",train_y )
print(train_x.shape)
valid_x=train_tweet[4000:]
valid_y=train_label[4000:]
print(" valid_x :",valid_x )
print(" valid_y :",valid_y )
test_x=test_text['text']
test_y=test_text['category']
print(test_x)
print(test_x.shape)
print(test_y)

encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
test_y = encoder.fit_transform(test_y)
print(train_y)
print(test_y)
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{3,}', max_features=1050)
#count_vect.fit(indic_tokenize.trivial_tokenize(indic_string))
count_vect.fit(train_tweet)

xtrain_count =  count_vect.transform(train_x)
xtest_count = count_vect.transform(test_x)
print("xtrain_count.shape  : ",xtrain_count.shape)
print("xtrain_count  : ",xtrain_count)
print("(train_x[0] : ",train_x[0])
print('x_train count is : ', xtrain_count[0])
count_vect.vocabulary_
def train_model(classifier, feature_vector_train, label, feature_vector_test, model_name):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    # predict the labels on testation dataset
    predictions = classifier.predict(feature_vector_test)
    pkl.dump(classifier, open(model_name,'wb'))
    cm=confusion_matrix(test_y, predictions)
    print("Confusion Matrix...:")
    print(cm)
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Greens')
    return metrics.accuracy_score(predictions, test_y), classification_report (predictions, test_y,digits=5)
# Naive Bayes on Count Vectors
hyper = {'alpha':[1,0.5],
         'fit_prior':[True, False]
        }
gd_nb=GridSearchCV(estimator=naive_bayes.MultinomialNB(),param_grid=hyper,verbose=True)
accuracy, classi = train_model(gd_nb, xtrain_count, train_y, xtest_count, 'NB_CV.pkl')
#accuracy, classi = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xtest_count, 'NB_CV.pkl')
print ("NB, Count Vectors: ", accuracy)
print ('\n',classi)
print(" best score:",gd_nb.best_score_)
print(" best estimator :",gd_nb.best_estimator_)
#Logitics regression
hyper = {'penalty':['l1','l2'],
         'solver':['lbfgs','liblinear']
        }
gd_lr=GridSearchCV(estimator=linear_model.LogisticRegression(),param_grid=hyper,verbose=True)
accuracy, classi = train_model(gd_lr, xtrain_count, train_y, xtest_count, 'LR_CV.pkl')
#accuracy, classi = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xtest_count, 'LR_CV.pkl')
print ("LR, Count Vectors: ", accuracy)
print ('\n',classi)
print(" best score:",gd_lr.best_score_)
print(" best estimator :",gd_lr.best_estimator_)

# SVM 

from sklearn.model_selection import GridSearchCV

hyper = {'C':[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1],
         'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
         'kernel':['rbf','linear']
        }

hyper = {'C':[0.05,0.1,0.2,0.3],
         'gamma':[0.1,0.2],
         'kernel':['rbf','linear']
        }

gd_svm=GridSearchCV(estimator=svm.SVC(),param_grid=hyper,verbose=True)

#accuracy, classi =train_model(svm.SVC(), xtrain_count, train_y, xtest_count, 'SVM_CV.pkl')
accuracy, classi =train_model(gd_svm, xtrain_count, train_y, xtest_count, 'SVM_CV.pkl')

print ("SVM, Count Vectors: ", accuracy)
print ('\n',classi)
print(" best score:",gd_svm.best_score_)
print(" best estimator :",gd_svm.best_estimator_)

# KNN 

hyper = { 'n_neighbors' : [15,20,25,30,35,40,45,50],
               'weights' : ['uniform','distance'],
               'metric' : ['minkowski','euclidean','manhattan']}

gd_knn=GridSearchCV(estimator=KNeighborsClassifier(n_neighbors=50),param_grid=hyper,verbose=True)
accuracy, classi =train_model(gd_knn, xtrain_count, train_y, xtest_count, 'KNN_CV.pkl')
print ("KNN, Count Vectors: ", accuracy)
print ('\n',classi)

print(" best score:",gd_knn.best_score_)
print(" best estimator :",gd_knn.best_estimator_)

