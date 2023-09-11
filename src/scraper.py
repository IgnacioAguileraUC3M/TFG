import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import selenium
from selenium.webdriver.common.by import By
from selenium.common import exceptions as selenium_exceptions
import chromedriver_autoinstaller
from lxml import etree
import src.exceptions as exceptions

class scraper:
    def __init__(self):
        pass

    def get(self, url:str) -> None:
        """Scraper goes to specified url"""
        pass

    def search_by(self, search_type, search_term, multiple_search) -> str|list:
        """Returns content found by searching according to given parameters
            search_type: attribute to use to search (CLASS, ID)
            search_term: query to search for
            multiple_search: wether to soerch for one item (False) or for all mathing the query (True)
        """
        pass

    @staticmethod    
    def filter_string(text:str, filter:list = [',', '"', ' ']) -> str:
        """Gets the given string and filters it"""
        text = text.encode("ascii", 'ignore').decode("utf-8", 'ignore')
        for item in filter:
            text = text.replace(item, ' ')
        for _ in range(100):
            text = text.replace('\n\n', '\n')
        return text


class requests_scraper(scraper):
    def __init__(self):
        super().__init__()
        self.HEADERS = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
        }

        self.soup = None
        self.search_options = ['CLASS', 'ID', 'XPATH', 'CUSTOM', 'TAG']

    def get(self, url: str):
        content = requests.get(url, headers=self.HEADERS).content
        self.soup = BeautifulSoup(content, 'html.parser', from_encoding="iso-8859-1")
    
    def get_page_text(self, filtered:bool = False) -> str:
        if filtered:
            text = self.filter_string(self.soup.text)
        else:
            text = self.soup.text
        return text

    def get_node(self, node):
        self.soup = BeautifulSoup(node.__str__(), 'html.parser')

    def search_by(self, search_type:str, search_term:str, multiple_search:bool = False, index:int = None, filtered:bool = False, tag_name:str = None, return_attr: str = None) -> str|list:
        if self.soup is None:
            raise exceptions.PageNotLoadedException
        
        if search_type not in self.search_options:
            raise exceptions.InvalidSearchOptionException(self.search_options)
        
        match search_type:
            case 'CLASS':
                if multiple_search:
                    node = self.soup.find_all(name=tag_name,attrs={"class": search_term})

                else: 
                    if index is None:
                        node = self.soup.find(name=tag_name,attrs={"class": search_term})

                    elif index == 'AUTO':
                        index = 0
                        has_text = False

                        try: 
                            while not has_text:
                                node = self.soup.find_all(name=tag_name,attrs={"class": search_term})[index]
                                txt = node.text
                                txt = txt.replace(' ', '')
                                txt = txt.replace('\n', '')
                                txt = self.filter_string(txt)

                                if len(txt) > 0:
                                    has_text = True
                                index += 1

                        except IndexError:
                            node = self.soup.find(name=tag_name,attrs={"class": search_term})
                    else:
                            node = self.soup.find(name=tag_name,attrs={"class": search_term})[index]


                if return_attr is None or return_attr == 'text':
                    if filtered:
                        if multiple_search:
                            texts = []

                            for n in node:
                                text = self.filter_string(n.text)
                                texts.append(text)
                            return texts
                        else:
                            try:
                                return self.filter_string(node.text)
                            except AttributeError:
                                return ''
                    else:
                        if multiple_search:
                            texts = []
                            for n in node:
                                texts.append(n.text)
                            return texts
                        else:
                            return node.text
                        
                elif return_attr == 'href':
                    if multiple_search:
                        href_list = []
                        for n in node:
                            href_list.append(n.get('href'))
                        return href_list
                    else:
                        return node.get('href')
                elif return_attr == 'node':
                    return node
                
            case 'ID':
                pass

            case 'TAG':
                node = self.soup.find_all(search_term)
                if return_attr == 'node':
                    return node
                elif return_attr is None or return_attr == 'text':
                    texts = []
                    for n in node:
                        if filtered:
                            texts.append(self.filter_string(n.text))
                        else:
                            texts.append(n.text)
                    return texts
                else:
                    attributes = []
                    for n in node:
                        attributes.append(n.get(return_attr))
                    return attributes

            case 'XPATH':
                dom = etree.HTML(str(self.soup))
                text = dom.xpath(f'{search_term}/text()')[0]
                if filtered:
                    return self.filter_string(text)
                else:
                    return text
            case 'CUSTOM':
                pass
        


class selenium_scraper(scraper):
    def __init__(self):
        chromedriver_autoinstaller.install()
        self.driver = webdriver.Chrome()
        super().__init__()

    def get(self, url: str) -> None:
        self.driver.get(url)
        return super().get(url)
    
    def search_by(self, search_type:str, serach_query:str, search_element=None):
        if search_element:
            driver = self.driver
            self.driver = search_element
        match search_type.lower():
            case 'class':
                element = self.driver.find_elements(By.CLASS_NAME, serach_query)

            case 'id':
                try:
                    element = self.driver.find_element(By.ID, serach_query)
                except selenium_exceptions.NoSuchElementException: return -1
            
            case 'tag':
                element = self.driver.find_elements(By.TAG_NAME, serach_query)

        if search_element:
            self.driver = driver
        return element


if __name__ == '__main__':
    scr = requests_scraper()
    scr.get('https://jointsdgfund.org//article/electronic-payment-life-changing-technology-social-protection-times-covid-19')
    search = scr.search_by('CLASS', 'field--name-field-text')
    search2 = scr.search_by('CLASS', 'field--name-body', index='AUTO', filtered=True)
    print(search2)