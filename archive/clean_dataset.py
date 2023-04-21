import os
def clean_dataset(path:str = None):
    if not path:
        path = './v1/data/dataset'
    # dataset_dir = os.listdir(path)
    # print(dataset_dir)
    for ods in range(1,18):
        ods_path = f'{path}/ODS{ods}'
        texts = os.listdir(ods_path)
        text_hashes = {}
        for text in texts:
            text_path = os.path.join(ods_path, text)
            with open(text_path,'r') as fp:
                text_content = fp.read()
            if len(text_content) < 10:
                os.remove(text_path)
                continue
            text_hashes[text_path] = hash(text_content)
        for t in text_hashes:
            t_hash = text_hashes[t]
            for j in text_hashes:
                if j != t:
                    j_hash = text_hashes[j]
                    if t_hash == j_hash:
                        try:
                            os.remove(j)
                            print(f'Text{j} removed')
                        except FileNotFoundError: pass
                        