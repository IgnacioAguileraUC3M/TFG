import os
import random
import pandas as pd

class dataset_manager():

    def __init__(self, dataset_path:str = './v1/data/dataset'):
        '''Class to manage dataset

            dataset_path: full or relative path to desired directory dataset, if
                null will automatically be set to ./v1/data/dataset
        '''

        self.dataset_path = dataset_path
        self.set_dataset()
       
    def ods_path(self,ods:int) -> str:
        '''Returns full directory path to specified ODS folder'''

        return f'{self.dataset_path}/ODS{ods}'

    def set_dataset(self) -> dict:
        '''Returns dictionary with all dataset entries'''

        dataset = {}
        for ods in range(1,18):
            ods_path = self.ods_path(ods)
            dataset[ods] = os.listdir(ods_path)

        self.dataset = dataset

    def get_last_entry(self, ods:int) -> str:
        '''Returns last entry i.e. textXX.txt with whigher XX for given ODS'''

        path = self.ods_path(ods)
        path_dir = os.listdir(path)

        try:
            path_dir.remove('desktop.ini')
        except ValueError:pass

        last_text_id = -1

        for entry in path_dir:
            text_id = int(entry[4:-4]) #textXX.txt
            if text_id >= last_text_id:
                last_text_id = text_id
        return f'text{last_text_id+1}.txt'

    def empty_dataset(self):
        '''Deletes all entries in dataset'''

        for ods in self.dataset:
            for text in self.dataset[ods]:
                os.remove(f'{self.dataset_path}/ODS{ods}/{text}')
        self.set_dataset()

    def add_entry(self, ods:int, text:str):
        '''Adds given text as string entry to given ODS'''

        path = self.ods_path(ods)
        entry_name = self.get_last_entry(ods)
        file_path = os.path.join(path,entry_name)

        with open(file_path, 'w') as fp:
            fp.write(text)

        self.set_dataset()

    def get_random_text(self,ods:int = None) -> str:
        '''Returns random text from given ODS, if no ODS 
        is specified it too will be chosen randomly'''
        if not ods:
            ods = random.randint(1,17)
            
        dataset = self.dataset[ods]
        ods_path = self.ods_path(ods)
        text_file = random.choice(dataset)

        with open(os.path.join(ods_path,text_file), 'r') as fp:
            text = fp.read()
        return f'ODS{ods}\nText:{text_file}\n {text}'
    
    def clean_dataset(self, min_characters:int = 10):
        '''Eliminates entries that are duplicate or have less than 
        the specified amount of characters, if not specified, min_characters 
        will be set to 10'''

        report = ''

        for ods in range(1,18):
            ods_path = self.ods_path(ods)
            texts = self.dataset[ods]
            text_hashes = {}

            for text in texts:
                text_path = os.path.join(ods_path, text)

                with open(text_path,'r') as fp:
                    text_content = fp.read()

                if len(text_content) < min_characters:
                    os.remove(text_path)
                    report += f'Removed {text_path} - reason: less than {min_characters} characters\n'
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
                                report += f'Removed {j} - reason: repeated text {t}\n'
                            except FileNotFoundError: pass
        self.set_dataset()
        print(report)


    def multi_label_augmentation(dataset:str='./all_t.csv', 
                                 augmentation:int=3, 
                                 interpolation:int = 4, 
                                 save_file:str = None,
                                 text_augment:bool = False):
        
        if text_augment:
            import nltk
            nltk.download('punkt')
            nltk.download('averaged_perceptron_tagger')
            nltk.download('wordnet')
            from textaugment import Wordnet
        
        dataset = pd.read_csv(dataset)
        # final_dataset = pd.DataFrame(columns = ['TEXT', 'CLASS'])
        final_dataset = pd.DataFrame(columns = ['TEXT', 'CLASS'])
        ds_length = len(dataset)
        SDGs = {}
        
        for i in range(1,18):
            SDGs[i] = []

        for _, element in dataset.iterrows():
            SDG_number = int(element['CLASS'][3:])  # SDGXX
            SDGs[SDG_number].append(element['TEXT'])
            element_label = element['CLASS']
            row = {
                'TEXT': element['TEXT'],
                'CLASS': str([element_label])
            }
            final_dataset = pd.concat([final_dataset, pd.DataFrame(row, index=[0])], ignore_index=True)


        for _ in range(ds_length*augmentation - ds_length):
            texts_chosen = random.randint(2, interpolation)
            sdgs_chosen = []
            final_text = ''

            for _ in range(texts_chosen):
                sdg = random.randint(1,17)
                while sdg in sdgs_chosen:
                    sdg = random.randint(1,17)
                    
                text = random.choice(SDGs[sdg])

                if text_augment:
                    verb = True
                    noun = True
                    t = Wordnet(v = verb, n = noun)
                    text = t.augment(text)

                final_text += f' {text}'
                sdgs_chosen.append(f'ODS{sdg}')
                
            row = pd.DataFrame({
                'TEXT': final_text,
                'CLASS': str(sdgs_chosen)
            },
            index = [0])

            final_dataset = pd.concat([final_dataset, row], ignore_index=True)

        if save_file:
            final_dataset.to_csv(save_file, index=False)
            return 0
        else:
            return final_dataset









if __name__ == '__main__':
    manager = dataset_manager()