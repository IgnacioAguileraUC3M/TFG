# Scrapes https://jointsdgfund.org/

import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from src import requests_scraper
from src import dataset_manager

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
 