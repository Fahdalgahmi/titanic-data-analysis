import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("titanic.csv")

# Normalize column names
df.columns = df.columns.str.lower()

print("Preview:")
print(df.head())

print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Fill missing values
df['age'] = df['age'].fillna(df['age'].mean())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# Create new column
df['family_size'] = df['sibsp'] + df['parch']

# Drop column with too many missing values
df = df.drop(columns=['cabin'])

print("\nBasic stats:")
print(df.describe())

print("\nSurvival count:")
print(df['survived'].value_counts())

print("\nSurvival by gender:")
print(df.groupby('sex')['survived'].mean())

print("\nSurvival by passenger class:")
print(df.groupby('pclass')['survived'].mean())

# Save cleaned file
df.to_csv("cleaned_titanic.csv", index=False)

# Chart 1: Age distribution
df['age'].hist()
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Chart 2: Survival rate by gender
df.groupby('sex')['survived'].mean().plot(kind='bar')
plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate")
plt.show()

print("\nDone. Cleaned file saved as cleaned_titanic.csv")