import csv
import pandas as pd
import numpy as np
import pandas_profiling

dataset = pd.read_csv('Workbook5Clean.csv')

pandas_profiling.ProfileReport(dataset)