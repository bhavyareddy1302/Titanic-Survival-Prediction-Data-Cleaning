import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("Titanic.csv")

# Display dataset information
print(df.info())
print(df.describe())

# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Encode categorical variables
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

# One-Hot Encoding
df = pd.get_dummies(df, columns=['Embarked'])

# Save cleaned dataset
df.to_csv("Titanic_Cleaned.csv", index=False)

print("Cleaned dataset saved successfully!")
