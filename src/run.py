try:
    import tensorflow_text
    import tensorflow_hub
except:
    pass
from keras.models import load_model
import tensorflow
from tensorflow.python.framework.errors_impl import InvalidArgumentError
from src.normalizer import normalizer, Summarizer
from src.scraper import requests_scraper
import numpy as np
import pandas as pd
from io import StringIO
import sys
import os
import json
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.autograph.set_verbosity(0)

class model_execution:
    '''Class creating an instance of a tf deep learning model'''

    def __init__(self, 
                 model_num:int, 
                 model_path:str = './models', 
                 checkpoint = False, 
                 threshold:float = 1, 
                 sigmoid:bool=False,
                 min_phrase_length:int = 3):
        '''
            :param: model_num: number indicating wich model to load 
            :param: model_path: directoiry in wich de specified model is located\
                                by default: ./models
            :param: checkpoint: Wether the model to load is the checkpoint of that model number
            :param: threshold: threshold to use when classifying new instances, by default 1
            :param: sigmoid: boolean attribute to indicate if the activation function of the model is sigmoid
            :param: min_phrase_lenght: not used
        '''

        if checkpoint:
            self.model_path = f'{model_path}/model{model_num}/checkpoint.tf'
        else:
            # self.model_path = f'{model_path}/model{model_num}.tf'
            self.model_path = f'{model_path}/model{model_num}/model{model_num}.tf'

        self.model_num = model_num
        self.model = load_model(self.model_path, compile=False)
        self.label_order = ["ODS1","ODS10","ODS11","ODS12","ODS13","ODS14","ODS15","ODS16","ODS17","ODS2","ODS3","ODS4","ODS5","ODS6","ODS7","ODS8","ODS9"]

        self.threshold = threshold #  if not sigmoid else 0.5
        self.min_phrase_length = min_phrase_length
        self.sigmoid = sigmoid
        self.normalizer = normalizer()



    def order_labels(self, predictions):
        SDGs_ordered = predictions.copy()
        SDGs_ordered.sort(reverse = True)

        # Creating a list of text labels sorted according to their probabilities
        for i, sdg in enumerate(self.label_order):
            value = predictions[i]
            j = SDGs_ordered.index(value)
            SDGs_ordered[j] = sdg

        predictions.sort(reverse=True)
        return SDGs_ordered, predictions


    def get_label_order(self, text:str) -> tuple[list[int], list[int]]:
        '''Gets a text and returns the labels sorted from most 
        probable to least probable along with the probabilities
        
            :param: text: string to clessify

            :returns: tuple containing the lebals sorted from most probable to least probable,\
                      along with their respective probabilities
        '''

        labels = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        phrases = 0
        if type(text) == list:
            batch_predictions = self.model.predict(np.array(text), verbose = 0).tolist()
            percentages = []
            classes_predicted = []
            for prediction in batch_predictions:
                prediction_sdgs, prediction_percentages = self.order_labels(prediction)
                classes_predicted.append(prediction_sdgs)
                percentages.append(prediction_percentages)
            return classes_predicted, percentages

        # Iterating through all the phrases in the given text
        for phrase in text.split('. '):

            # If phrase text is too small ignore it
            if len(phrase.split(' ')) < self.min_phrase_length:
                continue

            phrases += 1

            try:
                t_label = self.model.predict(np.array([phrase]), verbose=0).tolist()[0]
            except InvalidArgumentError: continue

            for i, l in enumerate(t_label):
                labels[i] += l 

        # Normalizing all the probailities between 0 and 1
        for i, l in enumerate(labels):
            if phrases:
                labels[i] = l/phrases

            # If no texts were classified set all probabilities to 0
            else:
                labels[i] == 0

        SDGs_ordered = labels.copy()
        SDGs_ordered.sort(reverse = True)

        # Creating a list of text labels sorted according to their probabilities
        for i, sdg in enumerate(self.label_order):
            value = labels[i]
            j = SDGs_ordered.index(value)
            SDGs_ordered[j] = sdg

        labels.sort(reverse=True)
        return SDGs_ordered, labels
    

    def get_prediction(self, text:str) -> tuple[tuple[(int)], tuple[(float)]]:
        '''Returns the labels predicted by the model and their corresponding probability
        
            :param: text: string to classify

            :returns: tuple containing both the labels classified by the model and\
                      their corresponding probability
        '''
        if type(text) == list:
            batch_predictions, batch_percentages = self.get_label_order(text)
            classifications = []
            classifications_percentages = []
            for predictions, percentages in zip(batch_predictions, batch_percentages):
                datum_sdgs = []
                datum_probabilities = []
                baseline = percentages[0]
                for sdg, probability in zip(predictions, percentages):
                    belonging_condition = probability >= self.threshold
                    if belonging_condition:
                        datum_sdgs.append(sdg)
                        datum_probabilities.append(probability)
                if len(datum_sdgs) == 0:
                    datum_sdgs.append(predictions[0])
                    datum_probabilities.append(percentages[0])
                
                classifications.append(datum_sdgs)
                classifications_percentages.append(datum_probabilities)
            return classifications, classifications_percentages

        text = str(text)
        predictions, percentages = self.get_label_order(text)
        SDGs = []
        probabilities = []
        baseline = percentages[0]

        # For every label and its probability assing by the model
        for sdg, probability in zip(predictions, percentages):

            if self.sigmoid:
                belonging_condition = probability >= self.threshold
            else:
                belonging_condition=  probability >= baseline*self.threshold

            if belonging_condition:
                SDGs.append(sdg)
                probabilities.append(probability)
                
        if len(SDGs) == 0:
            max_pred = max(percentages)
            SDGs.append(predictions[percentages.index(max_pred)])
            probabilities.append(max_pred)

        return (tuple(SDGs),tuple(probabilities))


    def run_test(self, 
                 test_data:str = './data/test_ds.csv', 
                 verbose:bool=False, 
                 save_report:str = None, 
                 summarized:bool=False):
        '''Classifies texts from given dataset and generates a 
        report containing the model accuracy and if selected 
        the classified texts and thir classifications
        
            :param: test_data: path to a csv containing a compatible dataset
            :param: verbose: boolean indicating wether to include test texts in the output
            :param: save_report: string indicating the path to\
                    the oputput file in wich to save the test report
            :param: summarized: boolean indicating if the output texts should be summarized or not
        '''

        test_dataset = pd.read_csv(test_data, encoding='cp1252', converters={'CLASS': pd.eval})
        correct = 0
        # total = len(test_dataset)
        total = 0
        report = ''
        predictions = {}
        n = normalizer()
        s = Summarizer()


        texts = []
        classes = []
        # For every occurence in the ds classify it and update the tp,fp and fn metrics
        for i, row in test_dataset.iterrows():
            text = str(row['TEXT'])

            if len(text) < 15:
                continue

            else:
                texts.append(text)
                classes.append(row['CLASS'])

        total_entries = len(texts)
        classifications, percentages = self.get_prediction(texts)

        tp = 0  # True Positives
        fp = 0  # False Positives
        fn = 0  # False Negatives

        for i, params  in enumerate(zip(classifications, classes, percentages)):
            classification = params[0]
            data_classes = params[1]
            data_percentages = params[2]

            for sdg in classification:
                sdg_num = sdg

                # If the label belongs to the text +1 to true positives
                if sdg_num in data_classes:
                    tp += 1
                
                # If the label does not belong to the text +1 to the false positives
                else:
                    fp += 1

            # For every label belonging to the text
            for sdg_num in data_classes:
                sdg = sdg_num

                # If the label is not classified by the model +1 to the false negatives
                if sdg not in classification:
                    fn += 1
                
            report += n.normalize_string(texts[i]) + '\n\n'
            report += f'Expected classes: {str(classes[i])}\n'
            report += f'Predicted classes: {str(classification)}\n'
            report += f'Predicted percentages: {str(data_percentages)}\n\n'
            report += f'=========================================\n'
            report += f'=========================================\n\n'


        try:
            precission = tp / (tp + fp)
        except ZeroDivisionError:
            precission = 0

        try:
            recall = tp / (tp + fn)
        except ZeroDivisionError:
            recall = 0

        try:
            f1_s = 2*((precission * recall) / (precission + recall))
        except ZeroDivisionError: 
            f1_s = 0

        tn = (total_entries*17)-(tp+fn+fp)

        accuracy = (tp+tn) / (tp + fp + tn + fn)

        report += f'Total test entries: {total_entries}\n'

        report += f'fp= {fp}\n'
        report += f'tp= {tp}\n'
        report += f'fn= {fn}\n'
        report += f'tn= {tn}\n'


        report += f'Precission= {precission}\n'
        report += f'Recall= {recall}\n'
        report += f'Accuracy= {accuracy}\n'
        report += f'F1 score= {f1_s}'

        if not save_report:
            print(report)
        else:
            with open(save_report, 'w', encoding='cp1252') as fp:
                fp.write(report)


    def classify_page(self, 
                      url, 
                      filtered:bool = False, 
                      save_file:str = None,
                      verbose:bool = False):
        '''Classifies the given page and returns the page text and its classification

                :param: url: link to the page to classify
                :param: filtered: indicates y page text should be filtered or not (by default it is not)
                :param: save_file: file in which to save the result
                :param: verbose: boolean indicating wether to include test texts in the output
                '''
        
        scraper = requests_scraper()
        scraper.get(url)
        page_text = scraper.get_page_text(filtered)

        if verbose:
            output = page_text
        else:
            output = ''
        output += '\n======================================\n\n'

        predictions, percentages = self.get_label_order(page_text)
        
        output += f'\nPage predictions: {predictions}'
        output += f'\nPage label percentages: {percentages}'

        if save_file:
            with open(save_file, 'w') as fp:
                fp.write(output)
        else: print(output)

        return output


    def classify_scopus_abstracts(self, 
                                  dataset:str, 
                                  save_file:str = None):
        '''Classifies all abstracs of given csv and returns the number of SDGs classified
                :param: dataset: scopus dataset
                :param: save_file: if specified, the result will be saved into said file, if not it will be printed'''
                
        # If abstracts come to WoS 
        if dataset[-3:] == 'xls': #.xls
            ds = pd.read_excel(dataset)
        else: ds = pd.read_csv(dataset)
        
        ds_size = len(ds)

        # Initializing SDG count dictionary to 0
        SDGs = {}
        for x in range(1,18):
            SDGs[f'ODS{x}'] = 0

        total_texts = 0
        total_classifications = 0
        multi_label = ''

        all_abstracts = list(ds['Abstract'].to_numpy())
        classes_predictes, all_percentages = self.get_prediction(all_abstracts)

        predicted_df = pd.DataFrame(columns=['TEXT','CLASSES', f'PREDICTIONS_{str(self.model_num)}'])

        # For every occurence classify it and add it to the report
        for i, text in enumerate(all_abstracts):
            try:
                text_len = len(text)
            except TypeError:
                continue
            if len(text) < 20:
                continue
            predictions = classes_predictes[i]
            percentages = all_percentages[i]

            # new_row = pd.DataFrame(
            #         {
            #             'TEXT' : text,
            #             'CLASSES': str(predictions),
            #             f'PREDICTIONS_{str(self.model_num)}' : str(percentages)
            #         },
            #         index = [0]
            #     )
            # predicted_df = pd.concat([predicted_df, new_row], ignore_index=True)
            total_texts += 1
            for sdg in predictions:
                SDGs[sdg] += 1
                total_classifications += 1
                
        SDGs['TOTAL TEXTS'] = ds_size
        SDGs['TOTAL CLASSIFIED'] = total_texts
        SDGs['TOTAL CLASSIFICATIONS'] = total_classifications

        if save_file:
            save_multi_label = save_file[:-4] + 'txt' # .json

            # with open(save_file, 'w') as fp:
            #     json.dump(SDGs, fp, indent=4)

            # with open(save_multi_label, 'w') as fp:
            #     fp.write(multi_label)
            predicted_df.to_csv(save_file, index=False)

            del(SDGs)
            del(ds)

        else: return SDGs

    def get_model_metrics(self, data:pd.DataFrame) -> tuple[float, float, float]: 
        '''Given a pandas dataframe containing multy label 
        ocuurences it classifies all of them and returns the 
        precission and recall metrics of the model in said dataset

            :param: data: compatible pandas dataframe containing the dataset

            :returns: tuple containing the precission, recall and accuracy of the model computed on the given dataset
        '''

        tp = 0  # True Positives
        fp = 0  # False Positives
        fn = 0  # False Negatives

        total_entries = len(data)
        texts = []
        classes = []
        # For every occurence in the ds classify it and update the tp,fp and fn metrics
        for i, row in data.iterrows():
            text = str(row['TEXT'])

            if len(text) < 15:
                continue

            else:
                texts.append(text)
                classes.append(row['CLASS'])

        classifications, _ = self.get_prediction(texts)
        # for aaa in classifications:
        #     print(aaa)
        # exit()
        # print(f'Classified texts: {len(classifications)}')
        # For every occurence classified by the model
        for classification, data_classes in zip(classifications, classes):
            for sdg in classification:
                # sdg_num = type(data_classes[0])(sdg[3:]) # SDGXX
                sdg_num = sdg
                # print(sdg_num)
                # print(data_classes)
                # If the label belongs to the text +1 to true positives
                if sdg_num in data_classes:
                    tp += 1
                
                # If the label does not belong to the text +1 to the false positives
                else:
                    fp += 1

            # For every label belonging to the text
            for sdg_num in data_classes:
                # sdg = f'ODS{sdg_num}'
                sdg = sdg_num


                # If the label is not classified by the model +1 to the false negatives
                if sdg not in classification:
                    fn += 1
                

        try:
            precission = tp / (tp + fp)
        except ZeroDivisionError:
            precission = 0

        try:
            recall = tp / (tp + fn)
        except ZeroDivisionError:
            recall = 0

        # print(classes)
        # print(classifications)
        # exit()
        # print(f'   ', end = '\r')
        tn = (total_entries*17)-(tp+fn+fp)
        # print(f'Text: {i}/{total_entries} | Precission: {precission} | Recall: {recall}\t\t\t\t\t', end = '\r')
        # print(f'\nTp: {tp}')
        # print(f'Tn: {tn}')
        # print(f'Fp: {fp}')
        # print(f'Fn: {fn}')
        accuracy = (tp+tn) / (tp + fp + tn + fn)
        return precission, recall, accuracy



    def tune_threshold(self, ds, iterations: int = 5, lr:float = 0.05):
        '''Given a dataset it tunes the threshold to optimize both, it precission and recall,
            incrementing or decreasing it by lr each iteration

            :param: ds: string indicating the path to the dataset to be used in the tuning 
            :param: iterations: number of times to compute the metrics and tune the threshold, by default 5
            :param: lr: float indicating the amount to vary the threshold in each iteration      
        '''

        data = pd.read_csv(ds, converters={'CLASS': pd.eval})

        for i in range(iterations):
            print(f'Iteration: {i}')
            print(f'Threshold: {self.threshold}')

            precission, recall = self.get_model_metrics(data)

            f1_s = 2*((precission * recall) / (precission + recall))

            print(f'Precission: {precission}')
            print(f'Recall: {recall}')
            print(f'F1 score: {f1_s}')
            
            tendency = precission/recall

            # Recall < precission -> more false negatives
            if tendency > 1:
                self.threshold -= lr

            # Recall > precission -> more false positives
            elif tendency < 1:
                self.threshold += lr

            elif tendency == 1:
                pass
            print(f'New threshold: {self.threshold}')
            print('-----------------')



    @staticmethod
    def get_text_joints(href:str, scraper:requests_scraper) -> str:
        scraper.get(href)
        texts = scraper.search_by('CLASS', 'field--item', multiple_search=True, filtered=True)
        text = ''

        for field in texts:
            text += f' {field}'
        return text
