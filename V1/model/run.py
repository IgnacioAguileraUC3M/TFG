from keras.models import load_model
from v1.modules.normalizer import normalizer
import numpy as np
import pandas as pd
from io import StringIO
import sys
import os

class model:
    def __init__(self, model_num:int, model_path:str = './v1/model/models', show_model_load = False):
        self.model_path = f'{model_path}/model{model_num}.tf'
        self.model = load_model(self.model_path, compile=False)
        self.label_order = ["ODS1","ODS10","ODS11","ODS12","ODS13","ODS14","ODS15","ODS16","ODS17","ODS2","ODS3","ODS4","ODS5","ODS6","ODS7","ODS8","ODS9"]
    def predict(self, text):
        label = self.model.predict(np.array([text]))
        i = np.argmax(label) 
        return self.label_order[i]
    
    def get_label_order(self, text):
        labels = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        phrases = 0
        for t in text.split('.'):
            if len(t.split(' ')) < 3:
                continue
            # print(t)
            phrases += 1
            try:
                t_label = self.model.predict(np.array([t])).tolist()[0]
            except: continue
            for i, l in enumerate(t_label):
                labels[i] += l 
            # print(self.label_order[np.argmax(t_label)])
        # label = self.model.predict(np.array([text])).tolist()[0]      
        SDGs_ordered = labels.copy()
        SDGs_ordered.sort(reverse = True)
        # for i, x in enumerate(self.label_order):
        #     SDGs_values[x] = label[i]
        for i, sdg in enumerate(self.label_order):
            value = labels[i]
            j = SDGs_ordered.index(value)
            SDGs_ordered[j] = sdg
            # print(i)
            # print(label)
            # SDGs_ordered = list(map(lambda x: x.replace(i, sdg), SDGs_ordered))
        # return 1
        labels.sort(reverse=True)
        return SDGs_ordered, labels
    
    def run_test(self, test_data:str = './v1/data/test_data.csv', verbose=False, save_file:str = None):
        test_dataset = pd.read_csv(test_data, encoding='cp1252')
        correct = 0
        total = len(test_dataset)
        report = ''
        predictions = {}
        n = normalizer()
        for index, row in test_dataset.iterrows():
            text = row['Text']
            expected_class = row['Class']
            predictions, percentages = self.get_label_order(text)
            first_prediction = int(predictions[0][3:])
            if first_prediction == int(expected_class):
                correct += 1
            if verbose:
                report += f'''{n.normalize_string(text)}

Expected class: {expected_class}
Predicted class: {first_prediction}
Ordered labels: {predictions}
Percentages labels: {percentages}
'''
            if index == total-1:
                report += '\n===============================\n\n'
            else:
                report += '\n--------------------------------\n'
        try:
            accuracy = (correct/total)*100
        except ZeroDivisionError:
            accuracy = 0
        report += f'Model test accuracy: {accuracy}%\n'
        report += f'Total test entries: {total}\n'
        report += f'Correctly classified entries: {correct}\n'
        report += f'Wrongly classified entries: {total-correct}'
        if not save_file:
            print(report)
        else:
            with open(save_file, 'w', encoding='cp1252') as fp:
                fp.write(report)



            


        

if __name__ == '__main__':
    mod = model(1)
    text = '''
    By 2030, significantly reduce the number of deaths and the number of people affected and substantially decrease the direct economic losses relative to global gross domestic product caused by disasters, including water-related disasters, with a focus on protecting the poor and people in vulnerable situations
    '''
    print(mod.predict(text))
    
    # data = pd.read_csv('./first_aprox/data/Test_data/SDG1/abstracts.csv')
    # data = pd.read_csv('./test_data.csv', encoding='cp1252')
    # titles = data['Title']
    # texts = data['Text']
    # texts = [text]

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__