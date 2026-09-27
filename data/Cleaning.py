import pandas as pd
df = pd.read_csv("data/raw/dataset.csv")
# 1. First 10 rows
print("----- HEAD -----")
print(df.head(10))
# 2. Rows and Columns
print("\n----- SHAPE -----")
print(df.shape)
# 3. Column Names
print("\n----- COLUMNS -----")
print(df.columns)
# 4. Dataset Information
print("\n----- INFO -----")
df.info()
print("\n----- DUPLICATES -----")
print(df.duplicated().sum())
print("\n----- DUPLICATE ROWS -----")
print(df[df.duplicated()].head())
