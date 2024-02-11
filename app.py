import numpy
import pandas
import pickle
import nltk
import streamlit as st
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
nltk.download('punkt')
nltk.download('stopwords')
porter = PorterStemmer()

model = pickle.load(open('model.pkl', 'rb'))
tf = pickle.load(open('vectorizer.pkl', 'rb'))

st.title('SMS classifier')
input = st.text_area('Enter the message')

def stemming(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(porter.stem(i))
    return ' '.join(y)

if st.button('Predict'):
    transformed_text = stemming(input)
    vectorize = tf.transform([transformed_text])
    result = model.predict(vectorize)

    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')
