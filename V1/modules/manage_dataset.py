import os
import random


class dataset_manager():
    def __init__(self, dataset_path:str = None):
        if not dataset_path:
            self.dataset_path = './v1/data/dataset'
        self.dataset = self.get_dataset()
       
    def ods_path(self,ods):
        return f'{self.dataset_path}/ODS{ods}'
    
    def get_dataset(self) -> dict:
        dataset = {}
        for ods in range(1,18):
            ods_path = self.ods_path(ods)
            dataset[ods] = os.listdir(ods_path)
        return dataset

    def get_last_entry(self, ods):
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
        for ods in self.dataset:
            for text in self.dataset[ods]:
                os.remove(f'{self.dataset_path}/ODS{ods}/{text}')

    def add_entry(self, ods, text):
        path = self.ods_path(ods)
        entry_name = self.get_last_entry(ods)
        file_path = os.path.join(path,entry_name)
        with open(file_path, 'w') as fp:
            fp.write(text)

    def get_random_text(self,ods:int = None)->str:
        if not ods:
            ods = random.randint(1,17)
            
        dataset = self.get_dataset()[ods]
        ods_path = self.ods_path(ods)
        text_file = random.choice(dataset)
        with open(os.path.join(ods_path,text_file), 'r') as fp:
            text = fp.read()
        return f'ODS{ods}\nText:{text_file}\n {text}'


if __name__ == '__main__':
    manager = dataset_manager()