import pandas as pd
import os 
import bokeh
from tabulate import tabulate
import random
from src import normalizer

DATA_PATH = './V1/data/dataset'
REPORT_TYPE = 'TABLE'
SAVE_OUTPUT = False
OUTPUT_PATH = './V1'
FILE_NAME = 'data_report.txt'
PRINT_OUTPUT = True
nr = normalizer()

def full_report(data_path:str = './V1/data/dataset',
                output_file:str = None) -> str:
    '''Generates a text report with the analysis of the dataset
    
        - data_path: dataset path inside directory, if not specified it is set to the default path 
        - output_file: file directory into wich save the report, if not specified the report will be printed
    '''

    report = ''
    classes = os.listdir(data_path)

    try:
        classes.remove('desktop.ini')
    except ValueError: pass

    table = []

    for class_ in classes:
        class_row = [class_]

        class_dir = os.listdir(data_path+'/'+class_)
        try:
            class_dir.remove('desktop.ini')
        except ValueError: pass

        class_len = len(class_dir)
        class_row.append(class_len)

        report +=   f'========={class_}=========\n'
        report +=   '=============================\n'
        report +=   f'Number of instances: {class_len}\n'

        avg_text_len = 0
        max_text_len = 0
        min_text_len = 10000000000000

        for dir in class_dir:
            with open(f'{data_path}/{class_}/{dir}', 'r') as fp:
                text = fp.read()

            avg_text_len +=  len(text)
            text_length  =   len(text)

            if text_length > max_text_len:
                max_text_len = text_length

            if text_length < min_text_len:
                min_text_len = text_length

        avg_text_len = avg_text_len//class_len

        report += f'''Average text length: {avg_text_len}
Max text length: {avg_text_len}
Minumum text length: {avg_text_len}\n'''

        random_text_name = random.choice(class_dir)

        with open(f'{data_path}/{class_}/{random_text_name}') as fp:
            random_text = fp.read()

        report +=   'Random text:\n-----------------------------\n'
        report +=   nr.normalize_string(random_text)
        report +=   '\n=============================\n\n'

    if output_file:
        with open(output_file, 'w') as fp:
            fp.write(report)
    else:
        print(report)

    return report



def table_report(data_path:str = './V1/data/dataset',
                 output_file:str = None) -> str:
    '''Generates a table report with the analysis of the dataset
    
        - data_path: dataset path inside directory, if not specified it is set to the default path 
        - output_file: file directory into wich save the report, if not specified the report will be printed
    '''

    HEADERS = ["CLASS", "NUMBER OF EXAMPLES", "AVERAGE TEXT LENGTH", "MAX TEXT LENGTH", "MIN TEXT LENGTH","NUMBER OF EMPTY TEXTS"]
    DATA_PATH = data_path

    total_entries = 0
    classes = os.listdir(DATA_PATH)
    try:
        classes.remove('desktop.ini')
    except ValueError: pass

    table = []

    for class_ in classes:
        class_row = [class_]
        number_of_zero_texts = 0
        class_dir = os.listdir(DATA_PATH+'/'+class_)
        try:
            class_dir.remove('desktop.ini')
        except ValueError: pass
        class_len = len(class_dir)
        total_entries += class_len

        if(class_len == 0):
            print('Empty dataset')
            return 'Empty dataset'
        
        avg_text_len = 0
        max_text_len = 0
        min_text_len = 10000000000000

        for dir in class_dir:
            with open(f'{DATA_PATH}/{class_}/{dir}', 'r') as fp:
                text = fp.read()

            avg_text_len += len(text)
            text_length = len(text)

            if text_length > max_text_len:
                max_text_len = text_length

            if text_length < min_text_len:
                min_text_len = text_length

            if text_length == 0:
                number_of_zero_texts += 1

        avg_text_len = avg_text_len//class_len


        class_row.append(class_len)
        class_row.append(str(avg_text_len))
        class_row.append(str(max_text_len))
        class_row.append(str(min_text_len))
        class_row.append(str(number_of_zero_texts))
        table.append(class_row)
    output = tabulate(table, headers=HEADERS) + f'\nTotal entries: {total_entries}'  

    if output_file:
        with open(output_file, 'w') as fp:
            fp.write(output)  
    else: print(output)
    return output
        