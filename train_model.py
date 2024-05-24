import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split # type: ignore
import pickle
from multinomial import MultinomialNB
from preprocessing import preprocess
from tfidf import TFIDF

# Load dataset
df = pd.read_csv('Dataset User Review OTA.csv')

# Preprocess the reviews
df['cleaned_review'] = df['content'].apply(preprocess)

# Split the data into training and testing sets (9:1 ratio)
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_review'], df['label'], test_size=0.1, random_state=42)

# Initialize the TFIDF transformer
tfidf_transformer = TFIDF(X_train.tolist())

# Transform the training data
X_train_tfidf = tfidf_transformer.create_tfidf_matrix()

# Transform the testing data
X_test_tfidf = tfidf_transformer.transform_batch(X_test.tolist())

# Initialize and train the MNB model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

# Calculate accuracy
accuracy = sum(y_pred == y_test) / len(y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save the trained model
with open('pickle/model_91_new.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('pickle/tfidf_transformer.pkl', 'wb') as transformer_file:
    pickle.dump(tfidf_transformer, transformer_file)