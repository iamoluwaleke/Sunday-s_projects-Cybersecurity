# Sunday's_Projects-Cybersecurity
# Cybersecurity_Machine Learning projects

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    'packets': [100, 2000, 150, 3000, 120],
    'duration': [2, 50, 1, 80, 3],
    'bytes': [500, 100000, 400, 200000, 600],
    'attack': [0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Features and labels
X = df[['packets', 'duration', 'bytes']]
y = df['attack']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Predictions:", predictions)
print("Accuracy:", accuracy)
