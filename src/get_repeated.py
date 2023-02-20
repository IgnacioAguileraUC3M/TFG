import json
with open('./odss_hres.json', 'r') as file:
    links = json.load(file)

enlaces = []
for x in links:
    enlaces += links[x]

# print(len(enlaces))
unique_links = list(set(enlaces))
repeated_links = {}

for enlace in unique_links:
    repeated_links[enlace] = []
for enlace in unique_links:
    for ods in links:
        if enlace in links[ods]:
            repeated_links[enlace].append(ods) 

with open('./repeticiones.json', 'w') as file:
    json.dump(repeated_links, file, indent=4)



    