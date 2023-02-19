# import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By


driver = webdriver.Chrome('./src/cromedriver.exe')


def google(query):
    url = f'https://www.google.com/search?client=chrome-b-d&q={query}'

    driver.get(url)

    urls_divs = driver.find_elements(By.CLASS_NAME, 'yuRUbf')
    results = []
    for div in urls_divs:
        a_elem = div.find_element(By.TAG_NAME, 'a')
        results.append(a_elem.get_attribute('href'))
    return results
    
for r in google('ods'):
    print(r)