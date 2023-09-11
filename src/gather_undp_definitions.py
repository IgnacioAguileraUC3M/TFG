from src import requests_scraper

def gather_data():
    base_xpath = '/html/body/div[1]/div/main/div/div/div[2]/article/div/div/div[2]/div[3]/div[1]/div[SDG]/div/div[2]/div/div/div[1]/div/'
    scraper = requests_scraper()
    scraper.get('https://www.undp.org/sustainable-development-goals')
    for x in range(1, 18):
        path = base_xpath.replace('SDG', str(x))
        for p in range(1,4):
            full_path = f'{path}p[{p}]'
            text = scraper.search_by('XPATH', full_path, filtered=True)
            add_text_to_test(text, x)
            
def add_text_to_test(text, ods):
    with open('./data/test_data.csv', 'r') as fp:
        test_data = fp.read()
    test_data += f'"{text}","{ods}"\n'
    with open('./data/test_data.csv', 'w') as fp:
        fp.write(test_data)