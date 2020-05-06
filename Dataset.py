import csv
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime



data = pd.read_csv("user-ct-test-collection-02.txt", sep="\t")




training = []
validation = []
test = []

train_limit = datetime(2006, 3, 5, 23, 59, 59)
valid_limit = datetime(2006, 3, 6, 23, 59, 59)
test_limit = datetime(2006, 3, 7, 23, 59, 59)
for i in range(len(data.index)):
    item = data.iloc[i]
    if not pd.isna(item.Query):
        date = datetime.strptime(item.QueryTime, "%Y-%m-%d %H:%M:%S")
        if (date <= train_limit):
            training.extend(item.Query.split(' '))
        elif (date <= valid_limit):
            validation.extend(item.Query.split(' '))
        elif date < test_limit:
            test.extend(item.Query.split(' '))





with open("training.txt", "wb") as f1:
    pickle.dump(training, f1)
    
with open("validation.txt", "wb") as f2:
    pickle.dump(validation, f2)

with open("test.txt", "wb") as f3:
    pickle.dump(test, f3)

