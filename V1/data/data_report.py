import pandas as pd
import os 
import bokeh
from tabulate import tabulate
import random
# from V1.modules.normalizer import normalizer
from v1.modules.normalizer import normalizer_class

def full_report():
    REPORT = ''
    classes = os.listdir(DATA_PATH)
    classes.remove('desktop.ini')
    table = []
    for class_ in classes:
        class_row = [class_]
        class_dir = os.listdir(DATA_PATH+'/'+class_)
        class_dir.remove('desktop.ini')
        class_len = len(class_dir)
        class_row.append(class_len)
        REPORT += f'========={class_}=========\n'
        REPORT+=   '=============================\n'
        REPORT += f'Number of instances: {class_len}\n'
        avg_text_len = 0
        max_text_len = 0
        min_text_len = 10000000000000
        counter = 0
        for dir in class_dir:
            counter += 1
            with open(f'{DATA_PATH}/{class_}/{dir}', 'r') as fp:
                text = fp.read()
            avg_text_len += len(text)
            text_length = len(text)
            if text_length > max_text_len:
                max_text_len = text_length
            if text_length < min_text_len:
                min_text_len = text_length
        avg_text_len = avg_text_len//counter
        REPORT+=f'''Average text length: {avg_text_len}
Max text length: {avg_text_len}
Minumum text length: {avg_text_len}\n'''
        random_text_name = random.choice(class_dir)
        with open(f'{DATA_PATH}/{class_}/{random_text_name}') as fp:
            random_text = fp.read()
        REPORT+= 'Random text-----------------------------\n'
        REPORT+=normalizer.normalize(74,random_text.split('\n'))
        REPORT+= '----------------------------------------\n'
        REPORT+=   '=============================\n\n'
    return REPORT



def table_report():
    OUTPUT = ''
    HEADERS = ["CLASS", "NUMBER OF EXAMPLES", "AVERAGE TEXT LENGTH", "MAX TEXT LENGTH", "MIN TEXT LENGTH","NUMBER OF EMPTY TEXTS"]
    empty_ds = True
    classes = os.listdir(DATA_PATH)
    classes.remove('desktop.ini')
    table = []
    for class_ in classes:
        class_row = [class_]
        number_of_zero_texts = 0
        class_dir = os.listdir(DATA_PATH+'/'+class_)
        try:
            class_dir.remove('desktop.ini')
        except ValueError: pass
        if(len(class_dir) != 0):
            empty_ds = False
            class_len = len(class_dir)
            avg_text_len = 0
            max_text_len = 0
            min_text_len = 10000000000000
            counter = 0
            for dir in class_dir:
                counter += 1
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
            avg_text_len = avg_text_len//counter
        else:
            class_len = 0
            avg_text_len = 0
            max_text_len = 0
            min_text_len = 0
            number_of_zero_texts = 0
        class_row.append(class_len)
        class_row.append(str(avg_text_len))
        class_row.append(str(max_text_len))
        class_row.append(str(min_text_len))
        class_row.append(str(number_of_zero_texts))
        table.append(class_row)
    OUTPUT+=tabulate(table, headers=HEADERS)        

    if not empty_ds:
        return OUTPUT
    else:
        return 'EMPTY DATASET'

def main():
    match REPORT_TYPE:
        case 'Table':
            out = table_report()
        case 'Full_report':
            out = full_report()
        case _:
            out = ''
        
    if PRINT_OUTPUT:
        print(out)
    if SAVE_OUTPUT:
        with open(f'{OUTPUT_PATH}/{FILE_NAME}', 'w') as fp:
            fp.write(out)
        

        
    

DATA_PATH = './first_aprox/data/dataset'
# DATA_PATH = './V1/data/dataset'
REPORT_TYPE = 'TABLE'
SAVE_OUTPUT = False
OUTPUT_PATH = './V1'
FILE_NAME = 'data_report.txt'
PRINT_OUTPUT = True
normalizer = normalizer_class()
