#import all the required libraries
import numpy as np
import pandas as pd
import pickle
from statistics import mode
import nltk
from nltk import word_tokenize
from nltk.stem import LancasterStemmer
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from tensorflow.keras.models import Model
from tensorflow.keras import models
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input,LSTM,Embedding,Dense,Concatenate,Attention
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup

#print("successfully imported everything")

# reading the dataset file from csv format 

data = pd.read_csv("Reviews.csv",nrows=100000)

# get rid of duplicate entries and rows with missing values 

data.drop_duplicates(subset=['Text'],inplace=True)

data.dropna(axis=0,inplace=True)

# single out input data

input_data = data.loc[:,"Text"]

# single out target data

target_data = data.loc[:,"Summary"]

# replace empty values with NaN 

target_data.replace('',np.nan,inplace=True)


# initialize variables

input_texts = []
target_texts = []
input_words = []
target_words = []

contractions=pickle.load(open("contractions.pkl","rb"))['contractions']

# getting stop words and the LancasterStemmer

stop_words = set(stopwords.words('english'))

stemm = LancasterStemmer()

def clean_input(text,src) : 
    
    #removing the html tags 

    text = BeautifulSoup(text,'html.parser').text 

    # tokenize the text into words 

    words = word_tokenize(text.lower())

    # filtering words which contains integers or a length smaller than 3 

    words = list(filter(lambda w: (w.isalpha() and len(w)>=3),words))

    #using the given contractions dictionary to deal with contracted words

    words = [contractions[w] if w in contractions else w for w in words]

    # stemming the words and filtering stop words

    if src == "inputs" : 
        words = [stemm.stem(w) for w in words if w not in stop_words]

    else : 
        words = [w for w in words if w not in stop_words]

    return words 



# passing the input records and taret records 

for in_txt, tr_txt in zip(input_data,target_data) : 

    in_words = clean(in_txt, "inputs")

    input_texts+= [' '.join(in_words)]

    input_words+= in_words

    #add 'sos' at the start and 'eos' at the end of text

    tr_words = clean("sos "+tr_txt+" eos","target")

    target_texts+=[' '.join(tr_words)]

    target_words+= tr_words


# storing only unique words from input and target list of words 

input_words = sorted(list(set(input_words)))

target_words = sorted(list(set(target_words)))

num_in_words = len(input_words)
num_tr_words = len(target_words)


max_in_len = (mode([len(i) for i in input_texts]))
max_tr_len = (mode([len(i) for i in target_texts]))


print("number of input words : ",num_in_words)
print("number of target words : ",num_tr_words)
print("maximum input length : ",max_in_len)
print("maximum target length : ",max_tr_len)


#split the input and target text into 80:20 ratio or testing size of 20%.
x_train,x_test,y_train,y_test = train_test_split(input_texts,target_texts
        ,test_size=0.2,random_state=0) 

# training the tokenizer with all the words 

in_tokenizer = Tokenizer() 
in_tokenizer.fit_on_texts(x_train)
tr_tokenizer = Tokenizer() 
tr_tokenizer.fit_on_texts(y_train)

#convert text into sequence of integers 
# where the integer will be the index of that word

x_train = in_tokenizer.texts_to_sequences(x_train)
y_train = tr_tokenizer.texts_to_sequences(y_train)


''' Keep going on DataFlair'''

