import requests
import streamlit as st

def api_endpoint(texte):
    url = '20.119.16.47/api/endpoint'      #'https://api-test-67.azurewebsites.net/api/endpoint'
    data = {'texte': texte}
    response = requests.post(url, data=data)
    return response #.text json()
    #st.session_state.user_output = response


st.title('Interface Streamlit pour API')

# Zone de texte pour l'entrée utilisateur
texte_utilisateur = st.text_area('Entrez votre texte ici', '')

# Bouton pour déclencher la requête à l'API
if st.button('Envoyer à l\'API'):
    if texte_utilisateur:
        # Appeler la fonction de l'API et afficher la réponse
        resultat_api = api_endpoint(texte_utilisateur)
        st.subheader('Réponse de l\'API')
        st.text_area("", value = resultat_api) #key="user_output")
        
        #st.json(resultat_api)
    else:
        st.warning('Veuillez entrer du texte avant d\'envoyer à l\'API.')


