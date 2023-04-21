# Scrapes https://jointsdgfund.org/

import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from v1.modules.scraper import requests_scraper
from v1.modules.manage_dataset import dataset_manager

ds_manager = dataset_manager()

def gather_data():
    url = 'https://jointsdgfund.org/'
    scraper = requests_scraper()
    scraper.get(url)
    sdgs_links = []
    for sdg in range(1,18):
        link = scraper.search_by('CLASS', 
                                 f"goal{sdg}", 
                                 tag_name='a', 
                                 multiple_search=False,
                                 return_attr='href')
        sdgs_links.append(link)
    for i, link in enumerate(sdgs_links,1):
        print(f'Scraping ODS{i}...', end="\r")
        scraper.get(url[:-1]+link)
        sdg_texts = scraper.search_by('CLASS',      # soup.find_all("div", {"class": "field--type-text-with-summary"})
                                      'field--type-text-with-summary',
                                      multiple_search=True,
                                      tag_name='div',
                                      filtered=True)
        
        targets_node = scraper.search_by('CLASS',      # soup.find_all("div", {"class": "field--type-text-with-summary"})
                                      'field--name-field-sdg-targets',
                                      return_attr='node')
        
        scraper.get_node(targets_node)
        target_texts = scraper.search_by('TAG',
                                         'li',
                                         filtered=True)
        
        sdg_texts += target_texts
        
        for text in sdg_texts:
            for t in split(['. ', '.\n', '\r\n'], text):
                if(len(t.strip()) < 20):
                    continue
                elif 'CONTACT US' in t or 'ABOUT' in t:
                    continue
                ds_manager.add_entry(i, t)
        
        
def split(delimiters, string, maxsplit=0):
    import re
    regex_pattern = '|'.join(map(re.escape, delimiters))
    return re.split(regex_pattern, string, maxsplit)
        # print(sdg_texts)
# exit()
# url = 'https://jointsdgfund.org/'
# headers = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
# }
# og_soup  = BeautifulSoup(requests.get(url, headers=headers).content, 'html.parser')

# sdgs_links = []
# for sdg in range(1,18):
#     sdg_a = og_soup.find("a", {"class": f"goal{sdg}"})
#     sdgs_links.append(url[:-1] + sdg_a.get('href'))

# def filter_string(text:str):
#     filters = [',', '"', "'", ' ']
#     for filter in filters:
#         text = text.replace(filter, '')
#         text = text.replace('\n', ' ')
#     return text



# dataset = pd.DataFrame(columns=['TEXT', 'ODS'])
# for link in sdgs_links:
#     content = requests.get(link, headers=headers).content
#     soup = BeautifulSoup(content, 'html.parser')
#     title = soup.find("span", {"class", "sdg-preline"}).text.replace("\n", "")
#     goal = title.split(' ')[1]
#     print(goal)
#     definition_block = soup.find_all("div", {"class": "field--type-text-with-summary"})
#     definition_text = definition_block[1].text
#     new_row = {'TEXT': filter_string(definition_text), 'ODS': str(goal)}
#     dataset = pd.concat([dataset, pd.DataFrame(new_row, index = [0])], ignore_index=True)
#     targets_block = soup.find("div", {"class": "field--name-field-sdg-targets"})
#     targets = targets_block.find_all("li")
#     for t in targets:
#         target_text = t.text
#         new_row = {'TEXT': filter_string(target_text), 'ODS': str(goal)}
#         dataset = pd.concat([dataset, pd.DataFrame(new_row, index = [0])], ignore_index=True)
    
# dataset.to_csv('./first_aprox/data/dataset_3.csv', index = False)