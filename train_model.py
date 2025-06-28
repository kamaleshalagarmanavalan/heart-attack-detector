import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv('CARDIO.csv')

# Features & target
X = df.drop(columns=['target'])
y = df['target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the model
model = BaggingClassifier()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'CARDIO.joblib')
print("✅ Model trained and saved as CARDIO.joblib!")
