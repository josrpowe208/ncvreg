import glob

import numpy as np
import pandas as pd

from src.ncvreg.models.ncvreg import NCVREG

X = np.random.randn(1000, 10)
y = np.random.randn(1000)

print(X.shape)
print(y.shape)

# Pick 4 random csv files from a directory and load then into 1 dataframe
df = pd.DataFrame()
files = glob.glob("/Users/jrp208/Documents/Independent Work/data/kaggle_stocks/stocks/*.csv")
# Randomly select n files from file
idxs = np.random.randint(0, int(len(files)), 4,)
for idx in idxs:
    file = files[idx]
    temp = pd.read_csv(file)
    # Add stock name to temp
    temp['symbol'] = file.split('.')[0].split('/')[-1]
    df = pd.concat([df, temp], axis=0)

print(df.describe())

X = df[-'adjusted_close']

model = NCVREG()
model.fit()
