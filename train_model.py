import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

# Load dataset
df = pd.read_csv('data/iris.csv')

# Features and labels
X = df.drop(['Id', 'Species'], axis=1)
y = df['Species']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Dictionary to store models
models = {
    'LogisticRegression': LogisticRegression(max_iter=200),
    'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=42)
}

best_model = None
best_accuracy = 0

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc}")
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

# Save the best model in the project folder
project_folder = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(project_folder, "model.pkl")

with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)

print(f"Best model saved as {model_path} ({type(best_model).__name__}) with accuracy {best_accuracy}")