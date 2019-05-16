import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#import data
df = pd.read_csv('Workbook5Clean.csv')

plt.style.use('classic')
fig = plt.figure()

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

#box plot of SOS of drafted and undrafted players
df_sosByDrafted = df[['SOS','Drafted']]

#subset data into drafted and undrafted
df_sos_undrafted = df.loc[df_sosByDrafted['Drafted'] == 0]
df_sos_drafted = df.loc[df_sosByDrafted['Drafted'] == 1]

#both scatter and box plots as subplots
fig = plt.figure()
ax = fig.add_subplot(221)
cax = ax.boxplot([df_sos_undrafted.SOS,df_sos_drafted.SOS])
ax.set_title('SOS')

#plt.subplot(2,2,1)
#plt.boxplot([df_sos_undrafted.SOS,df_sos_drafted.SOS])
#plt.ylabel('SOS')
#plt.title("SOS", fontsize=12)
#plt.xticks([1,2],['Undrafted','Drafted'])

plt.subplot(2,2,2)
plt.scatter(df['FG'],df['FGA'])
plt.title("FG vs FGA", fontsize=12)

plt.subplot(2,2,3)
plt.boxplot([df_sos_undrafted.USGp,df_sos_drafted.USGp])
plt.title("USGp", fontsize=12)
plt.xticks([1,2],['Undrafted','Drafted'])

plt.subplot(2,2,4)
plt.hist(df.TRB,20)
plt.title("TRB", fontsize=12)

plt.tight_layout()
plt.show()

fig.savefig('figures')

# find correlations between numerical values
corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')


#get column names
names = list(df.columns.values)
names.pop()
names.pop(0)
names.pop(0)
names.pop(0)

#show corrplot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,31,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names, rotation=90)
ax.set_yticklabels(names)
plt.show()
