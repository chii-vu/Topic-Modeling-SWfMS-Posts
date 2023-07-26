import pandas as pd
import re
from bertopic import BERTopic

new_df = pd.read_csv('Dataset/StackOverflowPostsDataset.csv')
new_df["merged"] = new_df[["Body", "Title", "Tags"]].apply("-".join, axis=1)

new_df.to_csv('Dataset/ConcatenatedDatasetSO.csv')

data = new_df.merged.values.tolist()

# Remove Emails
data = [re.sub('<[^<>]*>', '', sent) for sent in data]
# Remove new line characters
data = [re.sub('\s+', ' ', sent) for sent in data]
# Remove distracting single quotes
data = [re.sub("\'", "", sent) for sent in data]

# Remove stopwords
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

data = [[word for word in doc.split() if word not in stop_words] for doc in data]
data = [" ".join(doc) for doc in data]

# Train BERTopic on data
model = BERTopic(language="english", calculate_probabilities=True)
topics, probabilities = model.fit_transform(data)

# Get topics and their top words
topics = model.get_topic_freq()
topics.head()

# Visualize Topics
model.visualize_topics()
