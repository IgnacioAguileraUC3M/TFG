from src import selenium_scraper, requests_scraper
import time
import selenium
import pandas as pd

def gather_data():

    def get_links(sel, url):
        sel.get(url)

        load = True
        loads = 0
        max_loads = 250

        while load:
            load_button = sel.get_element('id', 'more_posts')
            if load_button == -1 or loads > max_loads:
                load = False
            else:
                try:
                    load_button.click()
                except:
                    time.sleep(2)
                    loads += 1
                    continue
        
            loads += 1
            time.sleep(0.5)

        articles_div = sel.get_element('id', 'ajax-posts')
        articles = sel.get_element('tag', 'a', articles_div)
        links = []
        for article in articles:
            links.append(article.get_attribute('href'))
        return links

    sel = selenium_scraper()
    article_hrefs = ['https://sdg.iisd.org/news/',
             'https://sdg.iisd.org/commentary/policy-briefs/',
             'https://sdg.iisd.org/commentary/guest-articles/',
             'https://sdg.iisd.org/commentary/generation-2030/']
    
    total_links = ''
    for href in article_hrefs:
        article_links = get_links(sel, href)
        for link in article_links:
            total_links += f'{link}\n'

    with open('./v1/temp/iisd_articles.csv', 'w') as fp:
        fp.write(total_links)

def scrape_link(link):
    req = requests_scraper()
    req.get(link)
    sdgs = []

    for i in range(1,18):
        sdg = req.search_by('CLASS', f'sdg sdg-{i} text -light -bold', multiple_search=True)
        if len(sdg):
            sdgs.append(i)

    if len(sdgs) == 0:
        return 0, 0

    content_nodes = req.search_by('CLASS', 'content', multiple_search=True, filtered=True)
    content = content_nodes[0].replace('\n', ' ')

    return content, sdgs

def get_links_data(links_path):
    with open(links_path, 'r') as fp:
        links = fp.read().split('\n')

    dataset = pd.DataFrame(columns = ['TEXT', 'CLASS'])
    total_links = len(links)
    for i, l in enumerate(links):
        print(f'Link {i}/{total_links}    ', end='\r')
        text, classes = scrape_link(l)
        if not text and not classes:
            continue
        
        new_row = pd.DataFrame(
            {
                'TEXT': text,
                'CLASS': str(classes)
            },
            index=[0]
        )
        dataset = pd.concat([dataset, new_row], ignore_index=True)

    dataset.to_csv('./v1/temp/iisd_dataset.csv', encoding='utf-8', index=False)