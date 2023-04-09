import requests
# from bs4 import BeautifulSoup
import json
from run import model
from scraper import requests_scraper
with open('./data/repeticiones_2.json', 'r') as fp:
    js = json.load(fp)

# HEADERS = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
# }

# def filter_string(text:str):
#     text = text.encode("ascii", 'ignore').decode("utf-8", 'ignore')
#     filters = [',', '"', ' ']
#     for filter in filters:
#         text = text.replace(filter, '')
#     return text

# def get_text_(href, scraprer):
#     response = requests.get(href, headers=HEADERS)
#     soup = BeautifulSoup(response.content, 'html.parser')
#     try:
#         description_node = soup.find_all(attrs={"class": "field--name-body"})[1]
#     except IndexError:
#         description_node = soup.find(attrs={"class": "field--name-body"})
#     try:
#         description_node_2 = soup.find_all(attrs={"class": "field--name-field-text"})[1]
#     except IndexError:
#         description_node_2 = soup.find(attrs={"class": "field--name-field-text"})
        
#     if description_node_2 is not None:
#         is_none = False
#         description_text = filter_string(description_node.text + description_node_2.text)
#     else:
#         is_none = True
#         description_text = filter_string(description_node.text)

#     if len(description_text) == 0:
#         print('text empty')
#     return description_text, is_none

def get_text(href, scraprer):
    scraper.get(href)
    texts = scraper.search_by('CLASS', 'field--item', multiple_search=True, filtered=True)
    text = ''
    for field in texts:
        text += field
    return text


if __name__ == '__main__':
    output = ''
    mod = model(1)
    scraper = requests_scraper()
    for x in js:
        if len(js[x]) == 1:
            text = get_text(x, scraper)
            prediction = mod.predict(text)
            original = js[x][0]
            output += (f'{x}----{text}\n--------------\nOriginal: {original}\nPrediction: {prediction}\n\n\t\t=====================================\n\n')

    with open('./out_test', 'w') as fp:
        fp.write(output)