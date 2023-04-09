import requests
from bs4 import BeautifulSoup
import selenium
import chromedriver_autoinstaller
import exceptions

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
        text = text.replace('\n\n', '\n')
        return text


class requests_scraper(scraper):
    def __init__(self):
        super().__init__()
        self.HEADERS = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
        }

        self.soup = None
        self.search_options = ['CLASS', 'ID']

    def get(self, url: str):
        content = requests.get(url, headers=self.HEADERS).content
        self.soup = BeautifulSoup(content, 'html.parser')

    def search_by(self,search_type:str, search_term:str, multiple_search:bool = False, index:int = None, filtered:bool = False) -> str|list:
        if self.soup is None:
            raise exceptions.PageNotLoadedException
        
        if search_type not in self.search_options:
            raise exceptions.InvalidSearchOptionException(self.search_options)
        
        match search_type:
            case 'CLASS':
                if multiple_search:
                    node = self.soup.find_all(attrs={"class": search_term})

                else: 
                    if index is None:
                        node = self.soup.find(attrs={"class": search_term})

                    elif index == 'AUTO':
                        index = 0
                        has_text = False

                        try: 
                            while not has_text:
                                node = self.soup.find_all(attrs={"class": search_term})[index]
                                txt = node.text
                                txt = txt.replace(' ', '')
                                txt = txt.replace('\n', '')
                                txt = self.filter_string(txt)

                                if len(txt) > 0:
                                    has_text = True
                                index += 1

                        except IndexError:
                            node = self.soup.find(attrs={"class": search_term})

                if filtered and not multiple_search:
                    return self.filter_string(node.text)

                elif not filtered and not multiple_search:
                    return node.text

                elif filtered and multiple_search:
                    texts = []

                    for n in node:
                        text = self.filter_string(n.text)
                        texts.append(text)
                    return texts
                
                elif not filtered and multiple_search:
                    texts = []
                    for n in node:
                        texts.append(n.text)
                    return texts
            
            case 'ID':
                pass
        


class selenium_scraper(scraper):
    def __init__(self):
        super().__init__()

if __name__ == '__main__':
    scr = requests_scraper()
    scr.get('https://jointsdgfund.org//article/electronic-payment-life-changing-technology-social-protection-times-covid-19')
    search = scr.search_by('CLASS', 'field--name-field-text')
    search2 = scr.search_by('CLASS', 'field--name-body', index='AUTO', filtered=True)
    print(search2)