import glob

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error

from src.ncvreg.models.ncvreg import NCVREG

# Pick 4 random csv files from a directory and load then into 1 dataframe
df = pd.DataFrame()
files = glob.glob("/Users/jrp208/Documents/Independent_Work/data/kaggle_stocks/stocks/*.csv")

# Randomly select n files from file
idxs = np.random.randint(0, int(len(files)), 4,)
for idx in idxs:
    file = files[idx]
    temp = pd.read_csv(file)
    if temp.isnull().values.any():
        continue
    temp.columns = [x.lower().replace(" ", "_") for x in temp.columns]
    # Add stock name to temp
    temp['symbol'] = file.split('.')[0].split('/')[-1]
    df = pd.concat([df, temp], axis=0)

X = df.drop(columns=['date', 'symbol', 'adj_close']).to_numpy()
y = df['adj_close'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = NCVREG(X_train, y_train)
model.fit()

res = model.predict(X_test)

print(root_mean_squared_error(y_test, res))
print(r2_score(y_test, res))
