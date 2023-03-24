import pandas as pd

dataset = pd.read_csv('./first_aprox/data/dataset_3.csv')

# print(len(dataset))
print(dataset.value_counts('ODS'))
# for i, data in enumerate(dataset['ODS']):
#     if str(data) == '13':
#         print(dataset['TEXT'][i])
#         print('\n')
#         print('================================')
#         print('\n')