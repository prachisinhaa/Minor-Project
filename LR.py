import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load the datasets
suspicious_df = pd.read_csv('/Users/prachisinha/Minor-Project/suspicious_accounts_dataset.csv')
genuine_df = pd.read_csv('/Users/prachisinha/Minor-Project/genuine_accounts_dataset.csv')

# Add labels to the datasets
suspicious_df['label'] = 1
genuine_df['label'] = 0

# Combine the datasets
combined_df = pd.concat([suspicious_df, genuine_df])

# Fill missing values
combined_df.fillna('', inplace=True)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(combined_df['title'] + ' ' + combined_df['text'])
y = combined_df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Function to predict if a post is suspicious
def predict_suspicious(title, text):
    post = title + ' ' + text
    post_vectorized = vectorizer.transform([post])
    prediction = model.predict(post_vectorized)
    return 'Suspicious' if prediction[0] == 1 else 'Genuine'

# Example usage
example_title = "Example post title"
example_text = "Example post text"
print("Prediction for example post:", predict_suspicious(example_title, example_text))