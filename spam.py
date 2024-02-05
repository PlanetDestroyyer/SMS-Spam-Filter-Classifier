import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
stopwords.words('english')
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def trans_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    l = []
    for i in text:
        if i.isalnum():
            l.append(i)
    text = l[:]
    l.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            l.append(i)
    text = l[:]
    l.clear()
    for i in text:
        l.append(ps.stem(i))
    return " ".join(l)

tfidf = pickle.load(open('tfidf.pkl','rb'))
mnb = pickle.load(open('mnb.pkl','rb'))

st.title('Spam Classifier Filter')

sms_input = st.text_input("Enter the Message")

if st.button("Predict"):
    transformed = trans_text(sms_input)

    vector = tfidf.transform([transformed])

    result = mnb.predict(vector)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
