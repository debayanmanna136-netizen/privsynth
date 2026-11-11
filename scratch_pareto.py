import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('c:/.vscode/privsynth/data/results/multi_objective_ctgan_results.csv', index_col=0)
scaler = MinMaxScaler()

fid = 1 - scaler.fit_transform(df[['Corr MAE', 'capital-gain KS', 'marital-status JSD', 'age KS']]).mean(axis=1)
util = scaler.fit_transform(df[['RF ROC-AUC', 'RF F1']]).mean(axis=1)
priv = scaler.fit_transform(df[['Syn->Real Mean Dist']]).flatten()

obj = np.column_stack((fid, util, priv))
pareto = np.ones(df.shape[0], dtype=bool)

for i, c in enumerate(obj):
    is_dominated = np.all(c <= obj, axis=1) & np.any(c < obj, axis=1)
    if np.any(is_dominated):
        pareto[i] = False

df['Fidelity'] = fid
df['Utility'] = util
df['Privacy'] = priv
df['Pareto'] = pareto
df['Overall'] = (fid + util + priv) / 3

print(df[['epochs', 'batch_size', 'embedding_dim', 'Privacy', 'Fidelity', 'Utility', 'Overall', 'Pareto']].round(4).to_string())
