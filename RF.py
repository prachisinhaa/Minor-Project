import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

# Define the model and pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),
    ('clf', RandomForestClassifier())
])

# Define the parameter grid for Grid Search
param_grid = {
    'clf__n_estimators': [100, 200, 300],
    'clf__max_depth': [None, 10, 20, 30],
    'clf__min_samples_split': [2, 5, 10],
    'clf__min_samples_leaf': [1, 2, 4]
}

# Perform Grid Search with Cross-Validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best model from Grid Search
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)

# Evaluate the model
print("Best Parameters:", grid_search.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Function to predict if a post is suspicious
def predict_suspicious(title, text):
    post = title + ' ' + text
    post_vectorized = vectorizer.transform([post])
    prediction = best_model.predict(post_vectorized)
    return 'Suspicious' if prediction[0] == 1 else 'Genuine'

# Example usage
example_title = "Example post title"
example_text = "Example post text"
print("Prediction for example post:", predict_suspicious(example_title, example_text))