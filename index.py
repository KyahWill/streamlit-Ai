import numpy as np 
import pickle 
import streamlit as st

# Load model
loaded_model = pickle.load(open('trained_model.sav',  'rb'))


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


def WFilterNonEnglish(sentence):
    new_sent = []
    for i in sentence.split():
        if d.check(i):
            new_sent.append(i)
    return new_sent


def PandasReadData(sentence):

    data = []
    data.append(WFilterNonEnglish(sentence))
    return data


def CreateAndTrainModel(data, size=300):
    model = gensim.models.Word2Vec(
        data, vector_size=size, window=10, min_count=1, workers=10)
    model.train(data, total_examples=len(data), epochs=10)
    return model
    # print(model.wv.most_similar(positive='sarcasm'))


def AverageVectorPerTweet(data, model):

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
            avg.append((np.average(row, axis=0)).tolist())
        except:
            avg.append(np.zeros((300,)).tolist())

    return avg


def ReadOpen(sentence):
    # sample = codecs.open(filename, "r", encoding="utf-8", errors="replace")
    # s = sample.read()

    data = []


    temp = []

    for j in word_tokenize(sentence):
        temp.append(j.lower())
        data.append(temp)

    print(data[1:])
    return data[1:]

def extract_emojis(sentence):
    print(sentence)
    return [word for word in sentence.split() if str(word.encode('unicode-escape'))[2] == '\\']


def CreateEmojiList(data):
    em = []
    for row in data:
        sentence = " ".join(row)
        em.append(extract_emojis(sentence))
    return em


def char_is_emoji(character):
    if character in emoji.distinct_emoji_list(character):
        return True
    else:
        return False


def FilterNonEnglish(sentence):

    ef = []
    for word in sentence:
        print("Word", word)
        if len(word) > 0:
            for c in word[0].split():
                print(c)
                if char_is_emoji(c):
                    ef.append(c)

    return ef


def GenerateEmojiVectors(emoji_list, pretrained_model):
    e2v = gensim.models.KeyedVectors.load_word2vec_format(
        pretrained_model, binary=True)

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
            Emoji_vec.append(np.average(row, axis=0).tolist())
        except:
            Emoji_vec.append(np.zeros((300,)).tolist())

    return Emoji_vec


def Emoji2Vec(sentence):
    # filename = "../Data/Final_Dataset_Word2Vec_Emoji2Vec.csv"
    print("Running Emoji2Vec...")
    data = ReadOpen(sentence)
    # data = PandasReadData(filename)
    # print(data[:10])
    temp_emojis = CreateEmojiList(data)
    print(temp_emojis)
    filterd_emojis = FilterNonEnglish(temp_emojis)
    print(filterd_emojis)
    # print(filterd_emojis[:10])
    Emoji_vec = GenerateEmojiVectors(filterd_emojis, "emoji2vec.bin")
    # print(Emoji_vec[:10])
    # print(len(Emoji_vec),len(data))
    return Emoji_vec


def Word2Vec(sentence):
    # filename = "../Data/Final_Dataset_Word2Vec_Emoji2Vec.csv"
    print("Running Word2Vec...")
    # data = ReadOpen(filename)
    data = PandasReadData(sentence)
    model = CreateAndTrainModel(data)
    avg = AverageVectorPerTweet(data, model)
    # print(avg)
    # print(len(data),len(avg))
    return avg


def JoinVectors(wv, ev, n):
    Concatenated_Vector = []
    size = 0
    if len(wv) > len(ev):
        size = len(ev)
    else:
        size = len(wv)
    for i in range(size):
        Concatenated_Vector.append(np.hstack((wv[i], ev[i])))

    return Concatenated_Vector


def Concat(string):
    word_vec = Word2Vec(string)
    print(len(word_vec))
    print("Word2vec done")
    Emoji_vec = Emoji2Vec(string)
    print(len(Emoji_vec))
    print("Emoji2vec Done")
    print("Concatenating...")
    Concatenated_Vector = JoinVectors(word_vec, Emoji_vec, len(word_vec))
    return Concatenated_Vector


def sentiment_analysis(input_data):
    input_value = Concat(input_data)
    print(input_value)
    return loaded_model.predict(np.array(input_value).reshape(1,-1))

def main():
    st.title('Sarcasm detection')

    detect = ''
    input_text = st.text_area(label="Input Tweeet",height=4,max_chars=140)
    if st.button("Detect Sarcasm"):
        detect = sentiment_analysis(input_text)
    st.success(detect)


if __name__ == "__main__":
    main()