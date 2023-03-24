# import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
# from keras.models import load_model
import numpy as np
import pandas as pd

with open('./data/repeticiones_2.json', 'r') as fp:
    js = json.load(fp)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
}



# model = load_model('./first_aprox/models/model_1.tf')
label_order = ["ODS1","ODS10","ODS11","ODS12","ODS13","ODS14","ODS15","ODS16","ODS17","ODS2","ODS3","ODS4","ODS5","ODS6","ODS7","ODS8","ODS9"]


def filter_string(text:str):
    text = text.encode("ascii", 'ignore').decode("utf-8", 'ignore')
    filters = [',', '"', ' ']
    for filter in filters:
        text = text.replace(filter, '')
    return text

def get_text(href):
    response = requests.get(href, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    try:
        description_node = soup.find_all(attrs={"class": "field--name-body"})[1]
    except IndexError:
        description_node = soup.find(attrs={"class": "field--name-body"})
    try:
        description_node_2 = soup.find_all(attrs={"class": "field--name-field-text"})[1]
    except IndexError:
        description_node_2 = soup.find(attrs={"class": "field--name-field-text"})
        
    if description_node_2 is not None:
        is_none = False
        description_text = filter_string(description_node.text + description_node_2.text)
    else:
        is_none = True
        description_text = filter_string(description_node.text)

    if len(description_text) == 0:
        print('text empty')
    return description_text, is_none


correct = 0
wrong = 0
out = ''
for x in js:
    if len(js[x]) == 1:
        text, is_none = get_text(x)
        # label = model.predict(np.array([text]))
    # print(label[0].index(max(label[0])))
        # print(label[0])
        # i = np.argmax(label)
        # prediction = label_order[i]
        # original = js[x][0]
        # if prediction == original:
        #     print(f'link {x} -> SDG{original} predicted as:')
        #     print(f'\tSDG: {prediction}')
        #     correct += 1
        # else:
        #     wrong += 1
        out += (f'{x}----{is_none}\n{text}\n\n\t\t=====================================\n\n')

with open('./out_test', 'w') as fp:
    fp.write(out)

# print(f'Correctly predicted: {correct}\nIncorrectly predicted: {wrong}\nTotal: {correct+wrong}')

