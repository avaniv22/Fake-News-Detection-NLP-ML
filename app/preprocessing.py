
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()                                         
    text = re.sub(r"http\S+|www\S+", "", text)                  
    text = re.sub(r"\d+", "", text)                             
    text = re.sub(r"[^\w\s]", "", text)                         
    tokens = nltk.word_tokenize(text)                           
    tokens = [w for w in tokens if w not in stop_words]         
    tokens = [lemmatizer.lemmatize(w) for w in tokens]          
    return " ".join(tokens)
