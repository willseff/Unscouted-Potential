import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#import data
df = pd.read_csv('Workbook5Clean.csv')

#turn certain columns into categories
df['Class'] = df['Class'].astype('category')
df['Pos'] = df['Pos'].astype('category')
df['Drafted'] = df['Drafted'].astype('category')

#subset data into positon and drafted
df_draftedByPosition = df[['Pos', 'Drafted']]
x = df_draftedByPosition.Pos
y = df_draftedByPosition.Drafted
table_draftedByPos = pd.crosstab(x,y, rownames = ['x'], colnames = ['y'])
print(table_draftedByPos)

#correlation between points and drafted (very trivial)
plt.scatter(df['FG'],df['FGA'])
plt.show()

#box plot of SOS of drafted and undrafted players
df_sosByDrafted = df[['SOS','Drafted']]

#subset data into drafted and undrafted
df_sos_undrafted = df_sosByDrafted.loc[df_sosByDrafted['Drafted'] == 0]
df_sos_drafted = df_sosByDrafted.loc[df_sosByDrafted['Drafted'] == 1]

#crea3te box plot

plt.boxplot([df_sos_undrafted.SOS,df_sos_drafted.SOS])
plt.show()

# find correlations between numerical values
corr = df.corr()

#get column names
names = list(df.columns.values)

#show corrplot
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
