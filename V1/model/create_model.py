import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import class_weight
from v1.modules.manage_models import new_model_name
from v1.execution.run import model
from matplotlib import pyplot as plt
import os

def new_model(seed:int = 1, 
              lr:float = 0.0001, 
              epochs:int = 10,
              ds_path:str = './v1/data/dataset', 
              model_path:str = './v1/model/models', 
              model_name:str = new_model_name(),
              checkpoint_path:str = f'./v1/model/models/{new_model_name()}/checkpoint.tf', 
              class_weights:bool = False, 
              architecture:str = 'recurrent',
              checkpoint:bool = True,
              shuffle:bool = False):


    os.mkdir(f'{model_path}/{model_name}/')
    train_dataset = tf.keras.utils.text_dataset_from_directory(ds_path, 
                                                            seed=seed,
                                                            labels = 'inferred', 
                                                            label_mode='categorical',
                                                            subset = 'training',
                                                            validation_split=0.2,
                                                            shuffle=shuffle)

    validation_dataset = tf.keras.utils.text_dataset_from_directory(ds_path, 
                                                            seed=seed,
                                                            labels = 'inferred', 
                                                            label_mode='categorical',
                                                            subset = 'validation',
                                                            validation_split=0.2,
                                                            shuffle=shuffle)
    
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                                    monitor='val_categorical_accuracy',
                                                                    mode='max',
                                                                    save_best_only=True)

    y_train = np.concatenate([y for x, y in train_dataset], axis=0)

    classes_list = []
    y_ints = [y.argmax() for y in y_train]

    class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                    classes=np.unique(y_ints),
                                                    y=y_ints)

    # print(class_weights)
    class_weight_dict = dict(enumerate(class_weights))
    # print(class_weight_dict)
    # exit()

    #----------------------------------------------------------
    #--------------------Decode labels-------------------------
    #----------------------------------------------------------
    # for i, label in enumerate(train_dataset.class_names):
    #   print("Label", i, "corresponds to", label)
    # exit()
    #----------------------------------------------------------
    #----------------------------------------------------------
    #----------------------------------------------------------

    train_text = train_dataset.map(lambda text, labels: text)

    VOCAB_SIZE = 1000
    encoder = tf.keras.layers.TextVectorization(
        max_tokens=VOCAB_SIZE, encoding='cp1252')
    encoder.adapt(train_text)

    #----------------------------------------------------------
    #------------------Get encoder vocab-----------------------
    #----------------------------------------------------------
    # vocab = np.array(encoder.get_vocabulary())
    # print(vocab)
    #----------------------------------------------------------
    #----------------------------------------------------------
    #----------------------------------------------------------

    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(
            input_dim=len(encoder.get_vocabulary()),
            output_dim=64,
            # Use masking to handle the variable sequence lengths
            mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(17, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                        loss='categorical_crossentropy',
                        metrics=['categorical_accuracy'])

    class_weight_dict = dict(enumerate(class_weights))
    history = model.fit(train_dataset, epochs=epochs,
                        validation_data=validation_dataset,
                        validation_steps=1, 
                        class_weight=class_weight_dict,
                        callbacks=[model_checkpoint_callback],
                        shuffle=shuffle)


    model.save(f'{model_path}/{model_name}/{model_name}.tf', save_format='tf')
    # run_model_test(model_name, model_path)
    create_history_report(history, epochs, f'{model_path}/{model_name}')

def create_history_report(history, epochs, path):
    epochs_stop=np.where(history.history['val_categorical_accuracy'] == np.max(history.history['val_categorical_accuracy']))
    report = f'''Total_epochs = {epochs}
Best epoch (checkpoint) = {epochs_stop[0][0]}
Final accuracy = {history.history['categorical_accuracy'][-1]}
Final val_accuracy = {history.history['val_categorical_accuracy'][-1]}
Checkpoint accuracy = {history.history['categorical_accuracy'][epochs_stop[0][0]]}
Checkpoint val_accuracy = {history.history['val_categorical_accuracy'][epochs_stop[0][0]]}
    '''
    with open(f'{path}/history_report.txt', 'w') as fp:
        fp.write(report)
    # Plot training accuracy
    train_g, = plt.plot(history.history['categorical_accuracy'], label='train')
    val_g, = plt.plot(history.history['val_categorical_accuracy'], label='val')
    plt.legend(handles=[train_g, val_g], loc='upper left')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.savefig(f'{path}/accuracy_plot.png')
    
def run_model_test(model_name, model_path):
    model_number = model_name[5:] # modelXX
    run_model = model(model_number)
    run_model.run_test(save_report=f'{model_path}/{model_name}/test_report.txt', verbose=True)
    run_model = model(model_number, checkpoint=True)
    run_model.run_test(save_report=f'{model_path}/{model_name}/checkpoint_report.txt', verbose=True)