import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv(r"C:\Users\HP\Downloads\Churn Prediction App\Telco-Customer-Churn.csv")
df

# Drop customerID, convert TotalCharges to numeric
df.drop("customerID", axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

# Encode categorical features
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

# Define X and y
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split data and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model and feature names
joblib.dump(model, "churn_model.pkl")
joblib.dump(X.columns.tolist(), "model_columns.pkl")