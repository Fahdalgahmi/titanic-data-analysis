import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("titanic.csv")

# normalize column names
df.columns = df.columns.str.lower()

print("Preview:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

# fill missing values
df['age'] = df['age'].fillna(df['age'].mean())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# create new column
df['family_size'] = df['sibsp'] + df['parch']

# drop useless column
df = df.drop(columns=['cabin'])

print("\nBasic stats:")
print(df.describe())

print("\nSurvival count:")
print(df['survived'].value_counts())

print("\nSurvival by gender:")
print(df.groupby('sex')['survived'].mean())

# visualization
df['age'].hist()
plt.title("Age Distribution")
plt.show()

# save cleaned file
df.to_csv("cleaned_titanic.csv", index=False)

print("\nDone. Cleaned file saved.")