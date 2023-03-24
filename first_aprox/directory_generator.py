import pandas as pd
import os

try: 
    os.mkdir('./first_aprox/data/dataset') 
except FileExistsError: pass

for i in range(1,18):
    try: os.mkdir(f'./first_aprox/data/dataset/ODS{i}') 
    except FileExistsError: pass

dataset = pd.read_csv('./first_aprox/data/dataset_3.csv')

for i in range(len(dataset)):
    text = dataset['TEXT'][i].encode('cp1252','ignore').decode("cp1252", "ignore")
    label = dataset['ODS'][i]
    directory = f'./first_aprox/data/dataset/ODS{label}/'
    directory_list = os.listdir(directory)
    try: directory_list.remove('desktop.ini')
    except ValueError: pass

    number = 0
    for item in directory_list:
        item_number = int(item[4:-4])+1
        if item_number > number:
            number = item_number
    name = f'text{number}.txt'
    with open(directory+name, 'w', encoding='cp1252') as fp:
        fp.write(text)


