from flask import Flask, request, jsonify
from flask_restful import reqparse, Api, Resource
import pickle
import pandas as pd
import numpy as np
import nltk

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
#api = Api(app)
    
filename = r'./data/cleaned/list_rare_words.csv'

list_rare_words = pd.read_csv(filename)
list_rare_words = list_rare_words['words_list'].to_list()

# In[3]:

filename = r'./model_saved/mlb_model'
mlb_model_loaded = pickle.load(open(filename, 'rb'))

# In[4]:

# loading TF-IDF pipe fitted model
filename = r'./model_saved//tf_idf_pipe_model'
tf_idf_model_loaded = pickle.load(open(filename, 'rb'))

# In[5]:

# OVR TF_IDF classification
filename = r'./model_saved/ovr_with_tf_idf_model'
ovr_with_tf_idf_model_loaded = pickle.load(open(filename, 'rb'))

# argument parsing
#parser = reqparse.RequestParser()
#parser.add_argument('query')


stop_words = set(nltk.corpus.stopwords.words('english'))

# In[7]:

def process_text(doc,
                   rejoin=True,
                   lemm_or_stemm="stem",
                   list_rare_words=None,
                   min_len_word=3,
                   force_is_alpha=True,
                   eng_words= None,
                   extra_words= stop_words) :
    
   
    """cf process_text_1 but with list_unique_words, min_len_word and force_is_alpha
    
    positional arguments :
    ----------------------
    doc : str : the document (aka a text in str format) to process
    
    opt args :
    ----------------------
    rejoin : bool : if True return a string else return the list of tokens - default = False
    lemm_or_stemm : str : if lem do lemmentize else stemmentize - default = stemmentize
    list_rare_words : list : a list of rare wrods to exclude - default = None
    min_len__word : int : the minimum length of wrods to not exlude - default = 3
    force_is_alpha : int : if 1, exclude all tokens with a numeric character - default = True
    eng_words : list : list of english words - default = None
    extra_words : list : exclude an extra list - default = None
    
    return :
    ----------------------
    a string (if rejoin is True) or a list of tokens
    """
    
    # list_unique_words
    if not list_rare_words:
        list_rare_words = []
        
    # lower
    doc = doc.lower().strip()
    
    # tokenize
    tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
    raw_tokens_list = tokenizer.tokenize(doc)
    
    # classics stop words
    cleaned_tokens_list = [w for w in raw_tokens_list if w not in stop_words]
    
    #####################################################
    #####################################################
    
    # no rare tokens
    non_rare_tokens = [w for w in cleaned_tokens_list if w not in list_rare_words]
    
    # no more len words
    more_than_N = [w for w in non_rare_tokens if len(w) >= min_len_word ]
    
    # only alpha chars
    if force_is_alpha :
        alpha_tokens = [w for w in more_than_N if w.isalpha()]
    else :
        alpha_tokens = more_than_N
        
    #####################################################
    #####################################################
    
    # stem or lem
    if lemm_or_stemm == "lem" :
        trans = nltk.stem.WordNetLemmatizer()
        trans_text = [trans.lemmatize(i) for i in alpha_tokens ]
    else:
        trans = nltk.stem.PorterStemmer()
        trans_text = [trans.stem(i) for i in alpha_tokens ]
    
    #####################################################
    #####################################################
    
    # in english
    if eng_words :
        engl_text = [i for i in trans_text if i in eng_words]
    else:
        engl_text = trans_text    
    
    #####################################################
    #####################################################
    
    # drop extra_words tokens
    if extra_words :
        final = [w for w in engl_text if w not in extra_words]
    else:
        final = engl_text
    
    #####################################################
    #####################################################
    
    # manage return type
    if rejoin :
        return " ".join(final)
    
 
    return trans_text



# argument parsing
#parser = reqparse.RequestParser()
#parser.add_argument('query',location=('args'))      #('json', 'query'))


@app.route('/api/endpoint', methods=['POST'])
def api_endpoint():
    #try:
        # Récupérer le texte envoyé en tant que données POST
    texte_req = request.form.get('texte')
    
    texte = np.array([texte_req])
    texte = pd.DataFrame(texte, columns=['text'])
    #Process text
    data_step1 = texte.text.apply(process_text, list_rare_words= list_rare_words)
    data_step1 = pd.DataFrame(data_step1, columns=['text'])
    # Classification step
    data_step2 = pd.DataFrame(tf_idf_model_loaded.transform(data_step1['text']))
    data_step3 = ovr_with_tf_idf_model_loaded.predict(data_step2)
    # Multilabelbinarizer inverse transform
    data_final = ", ".join(mlb_model_loaded.inverse_transform(data_step3)[0])
    
    #output = {'prediction': data_final}
    #return data_final
    #return jsonify({'texte': data_final})
    #return output
    return data_final #jsonify({"predictions" : data_final})

@app.route('/test', methods=['GET'])
def hello():
    data = {"data": "Hello World"}
    return jsonify(data)
    
if __name__ == '__main__':
    app.run(port=8000, debug=True)
    # app.run(host="0.0.0.0", debug=True, port=8080) # use_reloader=False
    
