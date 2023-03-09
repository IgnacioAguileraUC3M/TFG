from selenium import webdriver
import chromedriver_autoinstaller
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import json
from selenium.common.exceptions import NoSuchElementException
options = Options()
  
# this parameter tells Chrome that
# it should be run without UI (Headless)
# options.headless = True
chromedriver_autoinstaller.install()

url = 'https://eur-lex.europa.eu/'
driver = webdriver.Chrome()

search_term = 'sustainability'
year = 2022

query = f'./search.html?scope=EURLEX&text={search_term.replace(" ", "+")}&lang=en&type=quick&qid=1677691886608&DD_YEAR={year}&DTS_SUBDOM=LEGISLATION'

driver.get(url+query)

titles = driver.find_elements(By.CLASS_NAME, 'title')
documents_hrefs = []
for title in titles:
    documents_hrefs.append(title.get_attribute('href'))

for href in documents_hrefs:
    driver.get(href)
    has_summary = True
    celex = (driver.find_element(By.CLASS_NAME, 'DocumentTitle').text)[9:]
    document_type = ((driver.find_element(By.ID, 'title').text).split(' '))[0]
    try:
        summ_button = driver.find_element(By.LINK_TEXT, 'Document summary')
    except NoSuchElementException:
        has_summary = False
        continue
    summ_button.click()
    en_html = driver.find_element(By.ID, 'format_language_table_HTML_EN')
    driver.get(en_html.get_attribute('href'))
    aim = driver.find_element(By.TAG_NAME, 'body').text
    print(aim)
    # break
# /html/body/h2[4]

# print(len(titles))
# document_id = titles[0].get_attribute('href')[-19:-6]
# driver.get(f'https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=LEGISSUM:4591047&qid={document_id}')
# tab = driver.find_element(By.CLASS_NAME, 'legissumTab')

while 1:
    pass



