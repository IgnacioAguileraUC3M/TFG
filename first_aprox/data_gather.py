# Scrapes https://jointsdgfund.org/

# from selenium import webdriver
# import chromedriver_autoinstaller
# from selenium.webdriver.common.by import By
# import json
# chromedriver_autoinstaller.install()
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

url = 'https://jointsdgfund.org/'
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
}
og_soup  = BeautifulSoup(requests.get(url, headers=headers).content, 'html.parser')

sdgs_links = []
for sdg in range(1,18):
    sdg_a = og_soup.find("a", {"class": f"goal{sdg}"})
    # print(sdg_a)
    # continue
    # sdg_object = driver.find_element(By.CLASS_NAME, f'goal{sdg}')
    # sdgs_links.append(sdg_object.get_attribute('href'))
    sdgs_links.append(url[:-1] + sdg_a.get('href'))

# print(sdgs_links)
# field field--name-body field--type-text-with-summary field--label-hidden field--item

# dataset = pd.DataFrame(columns=['TEXT', 'ODS'])
# new_row = {'TEXT': str(description_text), 'ODS': str(ods)}
# dataset = pd.concat([dataset, pd.DataFrame(new_row, index = [0])], ignore_index=True)
def filter_string(text:str):
    filters = [',', '"', "'", ' ']
    for filter in filters:
        text = text.replace(filter, '')
        text = text.replace('\n', ' ')
    return text



dataset = pd.DataFrame(columns=['TEXT', 'ODS'])
for link in sdgs_links:
    content = requests.get(link, headers=headers).content
    soup = BeautifulSoup(content, 'html.parser')
    # title = soup.find("span", {"class", "sdg-preline"}).text.replace(" ", "")
    title = soup.find("span", {"class", "sdg-preline"}).text.replace("\n", "")
    goal = title.split(' ')[1]
    print(goal)
    definition_block = soup.find_all("div", {"class": "field--type-text-with-summary"})
    definition_text = definition_block[1].text
    new_row = {'TEXT': filter_string(definition_text), 'ODS': str(goal)}
    dataset = pd.concat([dataset, pd.DataFrame(new_row, index = [0])], ignore_index=True)
    targets_block = soup.find("div", {"class": "field--name-field-sdg-targets"})
    targets = targets_block.find_all("li")
    for t in targets:
        target_text = t.text
        new_row = {'TEXT': filter_string(target_text), 'ODS': str(goal)}
        dataset = pd.concat([dataset, pd.DataFrame(new_row, index = [0])], ignore_index=True)
    
dataset.to_csv('./first_aprox/data/dataset_3.csv', index = False)
# print('--------------------------------------------')
# print('--------------------------------------------')
# print(dataset)



# driver = webdriver.Chrome()
# driver.get(url)


# for link in sdgs_links:
#     driver.get(link)
#     text_block = driver.find_elements(By.CLASS_NAME, 'field--type-text-with-summary field--name-body')
#     print(len(text_block))

# with open('./data/dss_hrefs_2.json', 'w') as file:
#     json.dump(sdgs_links, file, indent=4)