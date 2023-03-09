import pandas as pd

dt = pd.read_csv('./dataset1.csv')
print(dt['ODS'].value_counts())