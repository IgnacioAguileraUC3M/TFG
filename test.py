import v1.data.data_report as data_report
# from v1.data.data_report import *
from v1.modules.manage_dataset import dataset_manager
from v1.data.gather_definitions import gather_data as gather_definitions
from v1.data.gather_reports_data import gather_data as gather_reports
from v1.model.create_model import new_model, run_model_test
from v1.execution.run import model
import pandas as pd
# from v1.data.gather_reports_data import gather_data
# dt = dataset_manager()
# dt.empty_dataset()
# gather_reports()
# gather_definitions()
# run_model_test('model6', './v1/model/models')
# texts = pd.read_csv('./archive/test_data.csv', encoding='cp1252')['Text']
# ml = model(2)
# ml.classify_page('https://www.unwomen.org/en/about-us/about-un-women', filtered = True)
# ml.run_test(verbose=True, test_data='./out_test.csv',save_file='./v1/model/test_report_articles_3.txt')
# ml.classify_scopus_abstracts('./archive/scopus(1).csv')
# exit()

# p = ml.predict('Global population is expected to cross 11 billion by the turn of the century, which has put immense pressure on the existing agricultural systems worldwide. This is complicated by gradually decreasing productivity and acreage as a result of climate change in addition to ever-increasing input costs of resource hungry staple crops like rice, wheat, and maize. Unfortunately, the most affected by these events are those who have the least resources at their disposal to mitigate the issue, especially in countries of Asia and Sub-Saharan Africa. It is therefore pertinent to explore and adopt alternative and/or complementary crops that are easier to cultivate, climate change tolerant, less resource hungry, nutritionally richer for human consumption, and agriculturally sustainable. Millets are perfect cereal crops which meet all of these requirements and can realistically provide much-needed solutions to current global food and nutritional security challenges. In this review, we provide a bird�s eye view of the relevance of millets in global agro-ecosystems in the context of their nutritional and agronomic attributes. Furthermore, we share perspectives on the major areas of crop improvement programs worldwide and discuss major challenges confronting the same. Finally, we discourse on the scope of millets for wider acceptability and highlight major points at the interface of genetic intervention�crop management post-harvest practices worth considering to potentially facilitate robust millet-based nutritional and food security. � 2022, The Author(s) under exclusive licence to Sociedad Chilena de la Ciencia del Suelo.')
# print(p)
# out = ''
# for _ in range(50):
#     out += dt.get_random_text() + '\n\n----------------------------\n\n'
# with open('./out.txt', 'w') as fp:
#     fp.write(out)
# dt.empty_dataset()
# dt.clean_dataset()
# p = data_report.table_report()
# print(p)
new_model(epochs=80, shuffle=True)
# dt.empty_dataset()
# gather_data()
# print(data_report.table_report())