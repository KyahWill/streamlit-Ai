from sklearn.metrics import f1_score,precision_score, recall_score
import matplotlib.pyplot as plt
import random
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import csv
import emoji
import numpy as np
import pickle
import enchant
import gensim
from gensim.models import Word2Vec
import pandas as pd
import nltk
from nltk import word_tokenize
import codecs
d = enchant.Dict("en_US")
# Emoji2vec


def extract_emojis(sentence):
	return [word for word in sentence.split() if str(word.encode('unicode-escape'))[2] == '\\']


def char_is_emoji(character):
	if character in emoji.distinct_emoji_list(character):
		return True
	else:
		return False


def text_has_emoji(text):
    for character in text:
        if character in emoji.distinct_emoji_list(character):
            return True
    return False

def ReadOpen(filename):
	# sample = codecs.open(filename, "r", encoding="utf-8", errors="replace")
	# s = sample.read()

    data = []

    with codecs.open(filename, 'r', encoding="utf-8", errors="replace") as readFile:
        reader = csv.reader(readFile)
        lines = list(reader)

    for i in lines:
        temp = []
        sentence = ' '.join(i)
        for j in word_tokenize(sentence):
            temp.append(j.lower())
            data.append(temp)
    
    print("data",data[1:])
    return data[1:]

def PandasReadData(filename):

	emoji_dataset=pd.read_csv(filename,index_col=False,encoding="ISO-8859-1")
	data = []

	for i in range(len(emoji_dataset)):
		data.append(emoji_dataset['Comments'][i].split())

	return data

def CreateEmojiList(data):
  em = []
  for row in data:
    sentence = " ".join(row)
    em.append(extract_emojis(sentence))

  return em

def FilterNonEnglish(em):

	new_em = []
	for row in em:
		ej = "".join(row)
		ef = []
		for c in ej:
			if char_is_emoji(c):
				ef.append(c)
		new_em.append(ef)

	return new_em

def GenerateEmojiVectors(emoji_list, pretrained_model):
	e2v = gensim.models.KeyedVectors.load_word2vec_format(pretrained_model, binary=True)

	Emoji_vec = []
	for i in range(len(emoji_list)):
		try:	
			row = []
			for j in emoji_list[i]:
				row.append(e2v[j])
			row = np.asarray(row)
			if len(row) < 1:
				Emoji_vec.append(np.zeros((300,)).tolist())
				continue
			Emoji_vec.append(np.average(row,axis=0).tolist())
		except:
			Emoji_vec.append(np.zeros((300,)).tolist()) 

	return Emoji_vec


def Emoji2Vec(filename):
	# filename = "../Data/Final_Dataset_Word2Vec_Emoji2Vec.csv"
	print("Running Emoji2Vec...")
	data = ReadOpen(filename)
	# data = PandasReadData(filename)
	# print(data[:10])
	temp_emojis = CreateEmojiList(data)
	filterd_emojis = FilterNonEnglish(temp_emojis)
	# print(filterd_emojis[:10])
	Emoji_vec = GenerateEmojiVectors(filterd_emojis,"emoji2vec.bin")
	# print(Emoji_vec[:10])
	# print(len(Emoji_vec),len(data))
	return Emoji_vec



def ReadOpen(filename):

    # sample = open("Final_Dataset_Word2Vec_Emoji2Vec.csv", "r",encoding = "ISO-8859-1")
    # sample = codecs.open(filename, "r", encoding="utf-8", errors="replace")
    sample = codecs.open(filename, "r", errors="replace") 
    sample = open(filename, "r") 
    s = sample.read() 

    f = s.replace("\n", " ") 
    data = []

    l = s.split('\n')
    for i in l:
        temp = [] 
            
        for j in word_tokenize(i):
            if d.check(j):
                temp.append(j.lower()) 
        
        data.append(temp)
    print(data)
    return data


def WFilterNonEnglish(sentence):
  new_sent = []
  for i in sentence.split():
    if d.check(i):
      new_sent.append(i)
  return new_sent



def PandasReadData(filename):
	emoji_dataset=pd.read_csv(filename,index_col=False,encoding="ISO-8859-1")
	data = []

	for i in range(len(emoji_dataset['Comments'])):
		data.append(WFilterNonEnglish(emoji_dataset['Comments'][i]))

	return data

def CreateAndTrainModel(data,size=300):
	model = gensim.models.Word2Vec(data,vector_size=size,window=10,min_count=2,workers=10)
	model.train(data, total_examples=len(data), epochs=10)
	return model
	# print(model.wv.most_similar(positive='sarcasm'))


def AverageVectorPerTweet(data,model):

	avg = []
	unused = []
	for i in range(len(data)):
		try:	
			row = []
			for j in data[i]:
				row.append(model[j])
			row = np.asarray(row)
			if len(row) < 1:
				avg.append(np.zeros((300,)).tolist())
				continue
			avg.append((np.average(row,axis=0)).tolist())
		except:
			avg.append(np.zeros((300,)).tolist())

	return avg


def Word2Vec(filename):
	# filename = "../Data/Final_Dataset_Word2Vec_Emoji2Vec.csv"
	print("Running Word2Vec...")
	# data = ReadOpen(filename)
	data = PandasReadData(filename)
	model = CreateAndTrainModel(data)
	avg = AverageVectorPerTweet(data,model)
	# print(avg)
	# print(len(data),len(avg))
	return avg


def JoinVectors(wv,ev,n):

	Concatenated_Vector = []

	for i in range(n):
		Concatenated_Vector.append(np.hstack((wv[i],ev[i])))
	
	return Concatenated_Vector

def ReadLabels(filename):
	ClassLabels = pd.read_csv(filename,index_col=False)

	return list(ClassLabels['Comments'])
	
def Concat(filename):
	word_vec = Word2Vec(filename)	
	print(len(word_vec))
	print("Word2vec done")
	Emoji_vec = Emoji2Vec(filename)
	print(len(Emoji_vec))
	print("Emoji2vec Done")
	print("Concatenating...")
	Concatenated_Vector = JoinVectors(word_vec,Emoji_vec,len(word_vec))
	return Concatenated_Vector

def RetrieveEmbeddings():
  filename = "Final_Dataset_Twitter.csv"
  print("Concatenating..." ) 
  Concatenated_Vector = Concat(filename) 
  return Concatenated_Vector


def ReadLabels(filename):
	ClassLabels = pd.read_csv(filename,index_col=False)

	return list(ClassLabels['Comments'])

def ML(X,Y):

    print("Training...")
    print(len(X))
    print(len(Y))
    X = np.array(X)
    Y = np.array(Y)
    # print(X)
	# for i in range(len(X)):
	# 	# print(i)
	# 	# print(X[i])
	# 	for j in X[i]:
	# 		j = float(j)

	# for i in Y:
	# 	for j in i:

	# 		j = float(j)


    kfold = KFold(10, shuffle=True, random_state=1)

    print(kfold)
    for train_index, test_index in kfold.split(X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
    print("success")

    # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    clf1 = svm.SVC(kernel = 'rbf')
    clf1.fit(x_train, y_train)
    predictions = clf1.predict(x_test)
    predictions[0:30]
    print("SVM RBF Kernel")
    rbf_score=f1_score(y_test, predictions, average = 'macro')
    recall =recall_score(y_test, predictions,)
    precision = precision_score(y_test,predictions,)
    print(rbf_score)
    print(recall)
    print(precision)
    return clf1

data = RetrieveEmbeddings()
labelfile = "Final_Dataset_Twitter_Labels.csv"
labels = ReadLabels(labelfile)
model = ML(data,labels)

print(model)

filename = "trained_model.sav"
pickle.dump(model, open(filename,'wb'))
