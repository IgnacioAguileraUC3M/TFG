from v1.modules.scraper import requests_scraper
from v1.modules.manage_dataset import dataset_manager
import re
import os
scraper = requests_scraper()
ds_manager = dataset_manager()

def get_report_texts(sdg, year):
    sdg_q = str(sdg).rjust(2, '0')
    scraper.get(f'https://unstats.un.org/sdgs/report/{year}/Goal-{sdg_q}/')
    text = scraper.search_by('CLASS','text-distribution', False, filtered=True)
    texts = scraper.search_by('CLASS', 'row', True, filtered=True)
    texts.append(text)
    print(f'Scraped SDG{sdg}, Year: {year}', end = '\r')
    return texts

def gather_data(dataset_path:str ='./V1/data/dataset'):
    for sdg in range(1, 18):
        for year in range(2016, 2023):
            texts = get_report_texts(sdg, year)
            for text in texts:
                dir_list = os.listdir(f'{dataset_path}/ODS{sdg}')
                try:
                    dir_list.remove('desktop.ini')
                except ValueError: pass
                max_num = 0
                for dir in dir_list:
                    n = int(dir.replace('text', '').replace('.txt', ''))
                    if n>max_num:
                        max_num = n
                # print(split(['. ', '.\n', '\r\n'], text))
                for t in split(['. ', '.\n', '\r\n'], text):
                    if(len(t.strip()) < 20):
                        continue
                    elif 'CONTACT US' in t or 'ABOUT' in t:
                        continue
                    ds_manager.add_entry(sdg, t)
                # final_dir = f'{dataset_path}/ODS{sdg}/text{max_num+1}.txt'
                # with open(final_dir, 'w') as f:
                #     f.write(text)

def split(delimiters, string, maxsplit=0):
    import re
    regex_pattern = '|'.join(map(re.escape, delimiters))
    return re.split(regex_pattern, string, maxsplit)