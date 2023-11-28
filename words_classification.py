
import streamlit as st
import requests


#from nltk.tokenize import RegexpTokenizer
#from nltk.corpus import stopwords


def on_click():
    # Text requested
    text_input = {"texte_input": st.session_state.user_input}
    
    #req = requests.post("http://127.0.0.1:8080/predict", texte_input= text_input)
    #resultat = req.json()
    #rec = resultat["predictions"]
    
    #st.session_state.user_output = rec
    st.session_state.user_output = text_input



st.title('Text classification') 

st.text_area("Text for classification", key="user_input", height= 100)
#st.text_input("Text for classification", key="user_input")
st.button("Classification", on_click=on_click)

st.text_input("Classification", key="user_output")


