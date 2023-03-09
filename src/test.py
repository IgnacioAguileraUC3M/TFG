import json

# with open('./dss_hrefs_2.json', 'r') as fp:
#     js = json.load(fp)

# with open('./dss_hrefs_2.1.json', 'w') as fp:
#     json.dump(js, fp, indent=4)

with open('./data/repeticiones_2.json', 'r') as fp:
    js = json.load(fp)


for x in js:
    if len(js[x]) == 1:
        print(f"href: {x}, ODS: {js[x]}")

exit()
import requests
from bs4 import BeautifulSoup

# The word to search for
word = "casa"

# Construct the URL for the search
url = f"https://dle.rae.es/{word}"

# Define headers to include in the request
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
}

# Send a GET request to the URL with the headers
response = requests.get(url, headers=headers)

# Parse the HTML content of the response using Beautiful Soup
soup = BeautifulSoup(response.content, "html.parser")

# get definitions
definitions = []
for p in soup.find_all("p", class_="j"):
    if p.find("span", class_="n_acep"):
        definition = ""
        for span in p.find_all("span"):
            if "data-id" in span.attrs:
                definition += span.text
        definitions.append(definition)

# print definitions
print(definitions)
