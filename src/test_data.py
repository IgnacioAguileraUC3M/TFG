import json
import requests
from bs4 import BeautifulSoup
import pandas as pd

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
}

def filter_string(text:str):
    filters = [',', '"', ' ']
    for filter in filters:
        text = text.replace(filter, '')
    return text
# ------------links 1------------
# with open('./data/odss_hres.json') as fp:
#     ods_hrefs = json.load(fp)

# dataset = pd.DataFrame(columns=['TEXT', 'ODS'])

# for ods in ods_hrefs:
#     for href in ods_hrefs[ods]:
#         href = href[0:19]+href[22:]
#         response = requests.get(href, headers=headers)
#         soup = BeautifulSoup(response.content, 'html.parser')
#         description_node = soup.find(attrs={"class": "text-formatted"})
#         description_text = filter_string(description_node.text)
#         new_row = {'TEXT': str(description_text), 'ODS': str(ods)}
#         dataset = pd.concat([dataset, pd.DataFrame(new_row, index = [0])], ignore_index=True)
# dataset.to_csv('./dataset.csv', index = False)
#------------------------------------


# ------------links 2------------
with open('./data/dss_hrefs_2.json') as fp:
    ods_hrefs = json.load(fp)
dataset = pd.read_csv('./dataset.csv')

for ods in ods_hrefs:
    ods_name = f'ODS{ods}'
    for i, href in enumerate(ods_hrefs[ods]):
        response = requests.get(href, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        try:
            description_node = soup.find_all(attrs={"class": "field--name-body"})[1]
        except IndexError:
            description_node = soup.find(attrs={"class": "field--name-body"})
            
        description_text = filter_string(description_node.text)
        if len(description_text) == 0:
            print('text empty')
            continue
        new_row = {'TEXT': str(description_text), 'ODS': str(ods_name)}
        dataset = pd.concat([dataset, pd.DataFrame(new_row, index = [0])], ignore_index=True)
        print(f'ods:{ods_name}, link:{i}', end = '\r')
dataset.to_csv('./dataset1.csv', index = False)
        

