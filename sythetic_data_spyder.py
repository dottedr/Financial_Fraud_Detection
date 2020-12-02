#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 17:01:49 2020

@author: sabinaadamska
"""

# source: https://www.kaggle.com/ntnu-testimon/paysim1
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("PS_20174392719_1491204439457_log.csv", na_values=['NA', '?'])
df.reindex(np.random.permutation(df.index))
small=df.head(10000)
small[:5]

encoder = LabelEncoder()
encoder.fit(small['type'])
encoded_type = encoder.transform(small['type'])

small.insert(1,"encoded_type",encoded_type)
small[:5]

columnsToEncode=small["type"].columns
small_enc = pd.get_dummies(small, columns=columnsToEncode, drop_first=True)
small_enc[:5]

mask = np.tril(small.corr())
sns.heatmap(small.corr(), annot = True, cmap= 'cool', mask=mask)



fig, ax = plt.subplots()

ax.hist(small_enc[small_enc["isFraud"]==1]["encoded_type"], bins=5, alpha=0.5, color="blue", label="fraud")
ax.hist(small_enc[small_enc["isFraud"]==0]["encoded_type"], bins=5, alpha=0.5, color="green", label="no fraud")

ax.set_xlabel("category")
ax.set_ylabel("Count transactions")

fig.suptitle("Fraud")

ax.legend();



fig, ax = plt.subplots()

scatter = ax.scatter(small_enc["encoded_type"], small_enc["isFraud"])