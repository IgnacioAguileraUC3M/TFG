from keras.models import load_model
import numpy as np
import pandas as pd

class model:
    def __init__(self, model_num):
        self.model = load_model(f'./first_aprox/models/model_{model_num}.tf', compile=False)
        self.label_order = ["ODS1","ODS10","ODS11","ODS12","ODS13","ODS14","ODS15","ODS16","ODS17","ODS2","ODS3","ODS4","ODS5","ODS6","ODS7","ODS8","ODS9"]
    def predict(self, text):
        label = self.model.predict(np.array([text]))
        i = np.argmax(label) 
        return self.label_order[i]   
        

if __name__ == '__main__':
    mod = model(3)
    text = '''
    By 2030, significantly reduce the number of deaths and the number of people affected and substantially decrease the direct economic losses relative to global gross domestic product caused by disasters, including water-related disasters, with a focus on protecting the poor and people in vulnerable situations
    '''
    # print(mod.predict(text))
    # data = pd.read_csv('./first_aprox/data/Test_data/SDG1/abstracts.csv')

    data = pd.read_csv('./test_data.csv', encoding='cp1252')
    # titles = data['Title']
    texts = data['Text']
    for t in texts:
        print(mod.predict(t))
    # texts = [text]