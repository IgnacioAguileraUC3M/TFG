from v1.data.clean_dataset import clean_dataset
import v1.data.data_report as data_report
# from v1.data.data_report import *
from v1.modules.manage_dataset import dataset_manager
from v1.data.gather_definitions import gather_data
dt = dataset_manager()
# out = ''
# for _ in range(50):
#     out += dt.get_random_text() + '\n\n----------------------------\n\n'
# with open('./out.txt', 'w') as fp:
#     fp.write(out)
# dt.empty_dataset()
p = data_report.table_report()
print(p)
# print(table_report())

# dt.empty_dataset()
# gather_data()