import requests
from googlesearch import search
from bs4 import BeautifulSoup
from bs4.element import Comment

query = 'ods'

pages = []

headers_Get = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:49.0) Gecko/20100101 Firefox/49.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }


def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)  
    return u" ".join(t.strip() for t in visible_texts)

def get_href(body):
    soup = BeautifulSoup(body, 'html.parser')
    a_tags = soup.findAll(name = 'a')
    refs = []
    for tag in a_tags:
        refs.append(tag['href'])
    return refs

# def get_links(link:str, links:list):
#     body = requests.get(link, headers=headers_Get).content
#     page_links = get_href(body)
#     links_to_parse = []
#     for link in page_links:
#         if link not in links:
#             links.append(link)
#             links_to_parse.append(link)
#     rec_links = []
#     for link in links_to_parse:
#         rec_links += get_links(link, links)
#     return links_to_parse

# print(get_links('https://www.un.org/sustainabledevelopment/es/objetivos-de-desarrollo-sostenible/', []))

search_results = list(search(query, tld="com", num=10, stop=10, pause=2))

page_content = requests.get(search_results[0], headers=headers_Get).content

# print(text_from_html(page_content))
page_text = ''

page_text += text_from_html(page_content) + '/n'
links = list(filter(lambda item: item is not None, get_href(page_content)))


for link in links:
        if link and link.startswith("http"):
            page_text += text_from_html(requests.get(link, headers=headers_Get).content)

filtered_text = page_text.replace("<undefined>", "")

# for link in links:
#     try:
#         page_text += text_from_html(requests.get(link, headers=headers_Get).content) + '/n'
#     except requests.exceptions.MissingSchema: pass

with open('output.txt', 'w') as file:
    file.write(filtered_text)

