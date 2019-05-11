import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Workbook5Clean.csv')

df['Class'] = df['Class'].astype('category')
df['Pos'] = df['Pos'].astype('category')
df['Drafted'] = df['Pos'].astype('category')

corr = df.corr()
print(corr)

#corrPlot = plt.matshow(df.corr())
#corrPlot.set_xticklabels(list(df.columns.values))
names = list(df.columns.values)

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,34,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names, rotation=90)
ax.set_yticklabels(names)
plt.show()
