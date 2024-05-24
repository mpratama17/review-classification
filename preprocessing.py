
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

def clean_text(text):
    # lower
    text = text.lower()
    # remove emot
    text = re.sub(r'\\u....', '', text)
    # remove url
    text = re.sub(r'http\S+', '', text)
    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # remove whitespace
    text = text.strip()
    return text

def remove_stopword(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = text.split()
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return ' '.join(filtered_text)

def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    word_tokens = text.split()
    lemmatized_text = [lemmatizer.lemmatize(word) for word in word_tokens]
    return ' '.join(lemmatized_text)

def case_folding(text):
    return text.lower()

def preprocess(text):
    text = clean_text(text)
    text = remove_stopword(text)
    text = case_folding(text)
    text = lemmatize(text)
    return text
# import re

# def preprocess(review):
#     review = review.lower()
#     review = re.sub(r'[^a-z\s]', '', review)
#     stopwords = set(['the', 'and', 'is', 'in', 'to', 'of', 'it', 'for'])
#     review = ' '.join([word for word in review.split() if word not in stopwords])
#     return review




if __name__ == '__main__':
    text = 'I am so riding :) https://www.google.com'    
    print(type(text)) # <class 'str'>
    print(clean_text(text)) # i am so happy
    print(remove_stopword(text)) # i happy
    print(lemmatize(text)) # I am so happy
    print(preprocess(text)) # i happy