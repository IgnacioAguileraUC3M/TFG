# Scrapes https://jointsdgfund.org/

from selenium import webdriver
import chromedriver_autoinstaller
import time
from selenium.webdriver.common.by import By
import json
chromedriver_autoinstaller.install()


url = 'https://jointsdgfund.org/'
driver = webdriver.Chrome()
driver.get(url)
# links = driver.find_elements(By.CLASS_NAME, 'sdg-overview-link')
# print(len(links))

# location.href='/article/greatest-risk-poverty-children-and-older-people'
def get_articles_links(driver):
    # articles = driver.find_elements(By.TAG_NAME, 'article')
    links = []
   
    last_page = driver.find_element(By.XPATH, "//a[@title='Go to last page']")
    last_page.click()
    time.sleep(2)
    page_a = driver.find_element(By.XPATH, "//a[@title='Current page']")
    page = int(page_a.text.split("\n")[1])
    while page > 1:
        divs = driver.find_elements(By.CLASS_NAME, 'image-wrapper')
        for div in divs:
            links.append('https://jointsdgfund.org/' + div.get_attribute('onclick')[15:-1])
        page -=1
        prev_page = driver.find_element(By.XPATH, f"//a[@title='Go to page {page}']")
        prev_page.click()
        time.sleep(2)
    divs = driver.find_elements(By.CLASS_NAME, 'image-wrapper')
    for div in divs:
        links.append('https://jointsdgfund.org/' + div.get_attribute('onclick')[15:-1])

    # for article in articles:
    #     div = article.find_element(By.CLASS_NAME, 'image-wrapper')
    #     links.append('https://jointsdgfund.org/' + div.get_attribute('onclick')[15:-1])
    return links
        


sdgs_links = {}
for sdg in range(1,18):
    object = driver.find_element(By.CLASS_NAME, f'goal{sdg}')
    driver.get(object.get_attribute('href'))
    links = get_articles_links(driver)
    sdgs_links[sdg] = links
    driver.get(url)
    # print(len(links))
    # print(links)

with open('./data/dss_hrefs_2.json', 'w') as file:
    json.dump(sdgs_links, file, indent=4)