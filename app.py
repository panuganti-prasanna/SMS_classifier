import streamlit as st
import string
punctuation = string.punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
import pickle
import sklearn


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))

model = pickle.load(open('model.pkl','rb'))

st.title('SMS_SPAM_CLASSIFIER')

input_sms = st.text_area('Enter the message')

if st.button('predict'):
    # 1. text preprocessing
    transformed_sms = transform_text(input_sms)
    # 2. vectorizing text
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. display
    if result == 1:
        st.header("SPAM")
    else:
        st.header("NOT SPAM")



