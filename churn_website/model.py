import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("../customer_churn_dataset_100k.csv")

# Drop CustomerID
df = df.drop("CustomerID", axis=1)

# Encode categorical columns
le = LabelEncoder()
for col in df.columns:
	if df[col].dtype == 'object':
		df[col] = le.fit_transform(df[col])

# Split features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

def predict_churn(data):
	return model.predict([data])[0]
