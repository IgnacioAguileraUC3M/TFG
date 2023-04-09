# # import selenium
# from selenium import webdriver
from selenium.webdriver.common.by import By


# driver = webdriver.Chrome('./src/cromedriver.exe')


# def google(query):
#     url = f'https://www.google.com/search?client=chrome-b-d&q={query}'

#     driver.get(url)

#     urls_divs = driver.find_elements(By.CLASS_NAME, 'yuRUbf')
#     results = []
#     for div in urls_divs:
#         a_elem = div.find_element(By.TAG_NAME, 'a')
#         results.append(a_elem.get_attribute('href'))
#     return results
    
# for r in google('ods'):
#     print(r)

from selenium import webdriver
import chromedriver_autoinstaller
import time
import json

chromedriver_autoinstaller.install()  # Check if the current version of chromedriver exists
                                      # and if it doesn't exist, download it automatically,
                                      # then add chromedriver to path

driver = webdriver.Chrome()
path = "https://sdgs.un.org/es/topics?n ame=&field_goals_target_id=All"

def get_links(driver):
    elements = driver.find_elements(By.CLASS_NAME, 'card-custom')
    hrefs = []
    for element in elements:
        href_elem = element.find_element(By.TAG_NAME, 'a')
        href = href_elem.get_attribute('href')
        hrefs.append(href)
    return hrefs


driver.get(path)
links = {}
for x in range(1,18):
    # time.sleep(1)
    selector = driver.find_element(By.ID, "edit-field-goals-target-id")
    for option in selector.find_elements(By.TAG_NAME, 'option'):
        if option.text == f"Objetivo {x}":
            option.click()
            driver.find_element(By.ID, 'edit-submit-topics-term').click()
            # time.sleep(3)
            links[f'ODS{x}'] = get_links(driver)
            break

print(links)
with open('./data/odss_hrefs.txt', 'w') as file:
    json.dump(links, file, indent=4)