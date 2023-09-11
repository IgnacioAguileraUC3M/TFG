import os

def new_model_name(models_path:str = './models'):
    models = os.listdir(models_path)
    try:
        models.remove('desktop.ini')
    except ValueError:
        pass
    
    try:
        models.remove('checkpoints.tf')
    except ValueError:
        pass
    
    model_number = 0
    if len(models) > 0:
        for mod in models:
            number = int(mod[5:]) #modelXX
            if number >= model_number:
                model_number = number
    return f'model{model_number+1}'

