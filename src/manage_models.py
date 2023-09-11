import os

def new_model_name(models_path:str = './models'):
    models = os.listdir(models_path)
    files_to_ignore = ['desktop.ini','checkpoints.tf','.gitkeep']
    for file in files_to_ignore:
        try:
            models.remove(file)
        except ValueError:
            pass
    
    model_number = 0
    if len(models) > 0:
        for mod in models:
            number = int(mod[5:]) #modelXX
            if number >= model_number:
                model_number = number
    return f'model{model_number+1}'

