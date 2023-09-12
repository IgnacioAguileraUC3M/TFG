# from src import ...
from src.create_model import new_model, new_bert_model
from src.run import model_execution

# new_bert_model()
new_model(epochs=5, lr = 1e-5, batch_size=5,architecture='bert', use_class_weights=False, checkpoint=True, shuffle=True)
# model_number = 45
# model_path = './models'
# model_name = 'model45'
# run_model = model_execution(model_number, sigmoid=True, threshold=0.5)
# run_model.run_test(save_report=f'{model_path}/{model_name}/test_report.txt', verbose=True)