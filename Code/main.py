import re
# import numpy as np 
# import pandas as pd
import pickle as pk
from nltk.stem import WordNetLemmatizer 

import warnings
warnings.filterwarnings("ignore") 

# models
from textblob import TextBlob

# Data Cleaning
def clean_text(text):  
    pat1 = r'@[^ ]+'                   # @signs and value
    pat2 = r'https?://[A-Za-z0-9./]+'  # links
    pat3 = r'\'s'                      # floating s's
    pat4 = r'\#\w+'                    # hashtags and value
    pat5 = r'&amp '
    pat6 = r'[^A-Za-z\s]'         #remove non-alphabet
    combined_pat = r'|'.join((pat1, pat2,pat3,pat4,pat5, pat6))
    text = re.sub(combined_pat,"",text).lower()
    return text.strip()

# Lemmatization
def tokenize_lem(sentence):
    lem = WordNetLemmatizer()
    outlist= []
    token = sentence.split()
    for tok in token:
        outlist.append(lem.lemmatize(tok))
    return " ".join(outlist)

# Sentiment Analysis Part 1
def SA1(text):                                          
    text = clean_text(text)                             #clean text
    text = tokenize_lem(text)                             #tokenize and lemmatize
    model = pk.load(open(r'pipeline_model.pkl', 'rb'))  #load pipeline model
    r1 = model.predict([text])[0]                         #predict sentiment 
    return r1, text

# Sentiment Analysis Part 2
def SA2(re1, text): 
    x = TextBlob(text).sentiment[0]                     #magnitude of  polarity
    x = abs(x)
    re2 = "Positive" if re1 > 0 else "Negative"
    if re1 == 0:
        print("Neutral")
    else:
        if 0 <= x < 0.3:
            print("Weakly {}".format(re2))
        elif 0.3 <= x < 0.65:
            print(re2)
        elif 0.65 <= x <= 1:
            print("Strongly {}".format(re2))


if __name__ == "__main__":
    ##### Input data #####
    ip = input("Enter text: ")

    ##### Output data #####
    op, cleaned_text = SA1(ip)
    SA2(op, cleaned_text)
    

