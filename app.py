import re  # Regular expression (provides additional info about this in ppt)
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords  # Stopword removal for words like 'in', 'a', 'an', etc.
from nltk.stem.porter import PorterStemmer  # PorterStemmer reduces words to their root form (e.g., hating -> hate)
from sklearn.feature_extraction.text import TfidfVectorizer  # Turn text into vector form
from sklearn.linear_model import \
    LogisticRegression  # Supervised learning used for classification problems (provides labels)
from sklearn.model_selection import train_test_split

# Read the dataset
news_df = pd.read_csv('C:/Users/anerd/Desktop/machine learning project fake news dataset/train.csv/train.csv')
news_df = news_df.fillna(' ')

# Create 'content' column by combining 'author' and 'title'
news_df['content'] = news_df['author'] + ' ' + news_df['title']

# Prepare X (features) and y (labels)
X = news_df.drop('label', axis=1)
y = news_df['label']

# Initialize PorterStemmer
ps = PorterStemmer()

# Stemming function
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)  # Remove non-alphabetic characters
    stemmed_content = stemmed_content.lower()  # Convert to lowercase
    stemmed_content = stemmed_content.split()  # Split into individual words
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]  # Stem words and remove stopwords
    stemmed_content = ' '.join(stemmed_content)  # Join the words back into a single string
    return stemmed_content

# Apply stemming function to the 'content' column
news_df['content'] = news_df['content'].apply(stemming)

# Vectorize the data
X = news_df['content'].values
y = news_df['label'].values
vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)

# Split the data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Fit the logistic regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Web interface
st.title('Fake News Detector')
input_text = st.text_input('Enter news article')

# Prediction function
def prediction(input_text):
    input_data = vector.transform([input_text])
    prediction = model.predict(input_data)
    return prediction[0]

if input_text:
    pred = prediction(input_text)
    if pred == 1:
        st.write('The news is fake')
    else:
        st.write('The news is real')
