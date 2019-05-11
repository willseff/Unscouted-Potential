import csv
import pandas as pd
import numpy as np

df = pd.read_csv('Workbook5Clean.csv')

df['Class'] = df['Class'].astype('category')
df['Pos'] = df['Pos'].astype('category')
df['Drafted'] = df['Pos'].astype('category')