import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle

# 1. Load and Clean data
print("Loading data...")
df = pd.read_csv('news.csv')
df = df.dropna(subset=['text']) # Removes empty rows

# 2. Prepare data
x = df['text']
y = df['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

# 3. Training (Simple version for small datasets)
print("Training the AI...")
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(x_train)

pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# 4. Save
pickle.dump(pac, open('model.pkl', 'wb'))
pickle.dump(tfidf_vectorizer, open('vectorizer.pkl', 'wb'))

print("SUCCESS: Model is now ready without errors!")