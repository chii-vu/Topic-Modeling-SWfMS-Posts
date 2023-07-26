import pandas as pd
import re
import nltk
from bs4 import BeautifulSoup
from bertopic import BERTopic

# Download stopwords if not already available
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def clean_text(text):
    # Remove HTML tags
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()

    # Remove non-alphanumeric characters and convert to lowercase
    text = re.sub(r'[^A-Za-z]', ' ', text.lower())

    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def remove_stopwords_and_stem(text):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    return " ".join(stemmer.stem(word) for word in text.split() if word not in stop_words)

def train_bert_topic(data):
    model = BERTopic(language="english", calculate_probabilities=True)
    topics, probabilities = model.fit_transform(data)
    return model, topics, probabilities

if __name__ == "__main__":
    # Load the data
    try:
        new_df = pd.read_csv('Dataset/StackOverflowPostsDataset.csv')
        new_df["merged"] = new_df[["Body", "Title", "Tags"]].apply("-".join, axis=1)
    except FileNotFoundError:
        print("Dataset file not found. Please provide the correct file path.")
        exit(1)

    # Preprocess the data
    new_df["merged"] = new_df["merged"].apply(clean_text)
    new_df["processed"] = new_df["merged"].apply(remove_stopwords_and_stem)

    # Save the preprocessed data
    new_df.to_csv('Dataset/ConcatenatedDatasetSO.csv', index=False)

    # Train BERTopic on processed data
    data = new_df["processed"].values.tolist()
    model, topics, probabilities = train_bert_topic(data)

    # Get topics and their top words
    topics_df = model.get_topic_freq()
    print(topics_df.head())

    # Save the BERTopic model
    model.save("bertopic_model")

    # Visualize Topics
    model.visualize_topics()
