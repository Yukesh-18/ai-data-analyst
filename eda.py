import pandas as pd

import pandas as pd

df = pd.read_csv("data.csv/data.csv.csv")

print(df.head())          # see the first 5 rows
print(df.info())          # see column types, missing values
print(df.describe())      # see min/max/average for numeric columns
print(df.isnull().sum())  # count missing values per column