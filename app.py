### streamlit
import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize Stemmer
ps=PorterStemmer()

# Text Preprocessing Function
def transform_text(text):
    text = text.lower()  # lower case
    text = nltk.word_tokenize(text)  # Tokenization

    y = []  # Removing special characters
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]  # Removing stop
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]  # stemming
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)
    
# Load vectorizer and model
tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

# Streamlit UI
st.title('Email/SMS Spam Classifier')

# Input box
input_sms=st.text_area ("Enter the message")

if st.button ('predict'):

    # preprocess
    transform_sms=transform_text((input_sms))

    # vectorizer
    vector_input=tfidf.transform([transform_sms])


    # predict
    result=model.predict(vector_input)[0]

    # display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")



### PyCharm Terminal
pip install streamlit
pip install NLTK
Python
>>> import nltk
>>> nltk . download('Punkt_tab')
>>> exit()
streamlit run app.py

