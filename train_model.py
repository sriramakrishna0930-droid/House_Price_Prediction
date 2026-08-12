import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("AmesHousing.csv")

# Select only 3 features
X = df[['Overall Qual', 'Gr Liv Area', 'Garage Cars']]

# Fill missing values
X = X.fillna(X.median())

# Target
y = df['SalePrice']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))

# Save model
with open("house_price_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")
