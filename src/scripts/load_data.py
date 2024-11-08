import glob

import numpy as np
import pandas as pd

from src.ncvreg.models.ncvreg import NCVREG

# Pick 4 random csv files from a directory and load then into 1 dataframe
df = pd.DataFrame()
files = glob.glob("/Users/jrp208/Documents/Independent_Work/data/kaggle_stocks/stocks/*.csv")

# Randomly select n files from file
idxs = np.random.randint(0, int(len(files)), 4,)
for idx in idxs:
    file = files[idx]
    temp = pd.read_csv(file)
    temp.columns = [x.lower().replace(" ", "_") for x in temp.columns]
    # Add stock name to temp
    temp['symbol'] = file.split('.')[0].split('/')[-1]
    df = pd.concat([df, temp], axis=0)

X = df.drop(columns=['date', 'symbol', 'adj_close']).to_numpy()
y = df['adj_close'].to_numpy()

model = NCVREG(X, y)
model.fit()

