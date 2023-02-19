# import requests
# from bs4 import BeautifulSoup

# headers_Get = {
#         'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:49.0) Gecko/20100101 Firefox/49.0',
#         'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
#         'Accept-Language': 'en-US,en;q=0.5',
#         'Accept-Encoding': 'gzip, deflate',
#         'DNT': '1',
#         'Connection': 'keep-alive',
#         'Upgrade-Insecure-Requests': '1'
#     }

# def get_text_from_url(url, visited_urls=set()):
#     visited_urls.add(url)
#     response = requests.get(url, headers=headers_Get)
#     soup = BeautifulSoup(response.text, "html.parser")
#     text = soup.get_text()
#     links = [link.get("href") for link in soup.find_all("a")]
#     for link in links:
#         if link and link.startswith("http") and link not in visited_urls:
#             text += get_text_from_url(link, visited_urls)
#     return text

# url = "https://www.un.org/sustainabledevelopment/es/objetivos-de-desarrollo-sostenible/"
# text = get_text_from_url(url)
# print(text)

import spacy

nlp = spacy.load("en_core_web_sm")

text = open("output.txt", "r").read()
doc = nlp(text)

for token in doc:
    print(token.text, token.pos_)