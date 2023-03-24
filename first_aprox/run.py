from keras.models import load_model
import numpy as np
import pandas as pd


model = load_model('./first_aprox/models/model_1.tf')
label_order = ["ODS1","ODS10","ODS11","ODS12","ODS13","ODS14","ODS15","ODS16","ODS17","ODS2","ODS3","ODS4","ODS5","ODS6","ODS7","ODS8","ODS9"]
text = '''
By 2030 achieve the sustainable management and efficient use of natural resources
'''
data = pd.read_csv('./first_aprox/data/Test_data/SDG1/abstracts.csv')
titles = data['Title']
# texts = data['Abstract']
texts = [text]
output = ''
for text in texts:
    label = model.predict(np.array([text]))
# print(label[0].index(max(label[0])))
    # print(label[0])
    i = np.argmax(label)
    output+=text+'\n'
    output += (label_order[i]) + '\n'
    output+= ('\n\t\t================================\n')
    print(output)

# with open('./output_2.txt', 'w', encoding='utf-8') as fp:
#     fp.write(output)