from scraper import requests_scraper
import os

def get_report_texts(sdg, year):
    sdg_q = str(sdg).rjust(2, '0')
    scraper.get(f'https://unstats.un.org/sdgs/report/{year}/Goal-{sdg_q}/')
    text = scraper.search_by('CLASS','text-distribution', False, filtered=True)
    texts = scraper.search_by('CLASS', 'row', True, filtered=True)
    texts.append(text)
    return texts


scraper = requests_scraper()
for sdg in range(1, 18):
    for year in range(2016, 2023):
        texts = get_report_texts(sdg, year)
        for text in texts:
            dir_list = os.listdir(f'./V1/data/dataset/ODS{sdg}')
            dir_list.remove('desktop.ini')
            max_num = 0
            for dir in dir_list:
                n = int(dir.replace('text', '').replace('.txt', ''))
                if n>max_num:
                    max_num = n
            final_dir = f'./V1/data/dataset/ODS{sdg}/text{max_num+1}.txt'
            with open(final_dir, 'w') as f:
                f.write(text)
            # print(f'SDG: {sdg} | Year: {year}\n-------------------------\n')
            # sdg_q = str(sdg).rjust(2, '0')
            # print(f'https://unstats.un.org/sdgs/report/{year}/Goal-{sdg_q}/')
            # print(text)

