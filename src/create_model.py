import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from sklearn.utils import class_weight
from src.manage_models import new_model_name
from src.run import model_execution
from matplotlib import pyplot as plt
import os
import random
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split


import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization


def new_model(seed:int = None, 
              lr:float = 0.0001, 
              epochs:int = 10,
              ds_path:str = './data/train_ds.csv', 
              model_path:str = './models', 
              model_name:str = new_model_name(),
              checkpoint_path:str = f'./models/{new_model_name()}/checkpoint.tf', 
              use_class_weights:bool = False, 
              architecture:str = 'recurrent',
              checkpoint:bool = False,
              shuffle:bool = False,
              batch_size:int=32):

    os.mkdir(f'{model_path}/{model_name}/')
    if seed is None:
        print('Random seed')
        seed = random.randint(1,1000)
        used_seed = f'{seed} (random)' 
    else: used_seed = seed

    # Loading and splitting thetraining data
    train_data = pd.read_csv(ds_path, converters={'CLASS': pd.eval})
    train_df, val_df = train_test_split(train_data,
                                        test_size = 0.2,
                                        shuffle=False)
    
    x_train = train_df['TEXT']
    x_val = val_df['TEXT']

    # Sepparating the classes, they will not be used like this
    y_train_raw = train_df['CLASS']
    y_val_raw = val_df['CLASS']

    # Transformming the classes  into a binary vector
    encoder = MultiLabelBinarizer()
    y_train = encoder.fit_transform(y_train_raw)
    y_val = encoder.fit_transform(y_val_raw)
    # print(y_train)
    # print('----')
    # print(y_val)
    train_text = x_train.tolist()
    steps_per_epoch = len(x_train)


    if checkpoint:
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                                        monitor='val_binary_accuracy',
                                                                        mode='max',
                                                                        save_best_only=True)
        callbacks = [model_checkpoint_callback]
    else:
        callbacks = None

    if use_class_weights:
        # Calculating the class weights to be used, if required, in the training 
        y_ints = [y.argmax() for y in y_train]
        class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                        classes=np.unique(y_ints),
                                                        y=y_ints)
        class_weight_dict = dict(enumerate(class_weights))
    else:
        class_weight_dict = None


    #----------------------------------------------------------
    #--------------------Decode labels-------------------------
    #----------------------------------------------------------
    # for i, label in enumerate(train_dataset.class_names):
    #   print("Label", i, "corresponds to", label)
    # exit()
    #----------------------------------------------------------
    #----------------------------------------------------------
    #----------------------------------------------------------


    match architecture:
        case 'recurrent':
            model = build_recurrent_model(train_text)
        case 'bert':
            model = build_bert_model()
            if epochs > 15: 
                print('Epochs maybe too high to train BERT')
                print(f'\tEpochs= {epochs}')
                print('===============')
            if batch_size > 15: 
                print('Batch size maybe too high to train BERT, if using a slow computer downsize it')
                print(f'\tBatch size= {batch_size}')
                print('===============')


    loss = keras.losses.BinaryCrossentropy()
    metrics = tf.metrics.BinaryAccuracy()

    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1*num_train_steps)

    init_lr = lr
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')
    
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

    history = model.fit(x_train,
                        y_train,
                        validation_data=(x_val, y_val),
                        epochs=epochs,
                        class_weight=class_weight_dict,
                        callbacks=callbacks,
                        # shuffle=shuffle,
                        batch_size=batch_size)

    params = ''
    params += f'Architecture= {architecture}\n'
    params += f'Epochs= {epochs}\n'
    params += f'Learning rate= {lr}\n'
    params += f'Class weights= {use_class_weights}\n'
    params += f'Checkpoint= {checkpoint}\n'
    params += f'Shuffle= {shuffle}\n'
    params += f'Batch size= {batch_size}\n' 
    params += f'Seed= {used_seed}'

    model.save(f'{model_path}/{model_name}/{model_name}.tf', save_format='tf')

    with open(f'{model_path}/{model_name}/params.txt', 'w') as fp:
        fp.write(params)

    create_history_report(history, epochs, f'{model_path}/{model_name}')
    run_model_test(model_name, model_path, checkpoint=checkpoint)
    

def build_recurrent_model(train_text):
    VOCAB_SIZE = 2000
    encoder = keras.layers.TextVectorization(
        max_tokens=VOCAB_SIZE)
    encoder.adapt(train_text)

    #----------------------------------------------------------
    #------------------Get encoder vocab-----------------------
    #----------------------------------------------------------
    # vocab = np.array(encoder.get_vocabulary())
    # print(len(vocab.tolist()))
    #----------------------------------------------------------
    #----------------------------------------------------------
    #----------------------------------------------------------
    model = keras.Sequential([
        encoder,
        keras.layers.Embedding(
            input_dim=len(encoder.get_vocabulary()),
            output_dim=128,
            # Use masking to handle the variable sequence lengths
            mask_zero=True),
        keras.layers.Bidirectional(keras.layers.LSTM(128)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(17, activation=keras.activations.sigmoid)
    ])
    return model


def build_bert_model():
  preprocess_model = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
  encoder_model = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/2'

  encoder               =   hub.KerasLayer(encoder_model, trainable=True, name='BERT_encoder')
  preprocessing_layer   =   hub.KerasLayer(preprocess_model, name='preprocessing')

  text_input            =   keras.layers.Input(shape=(), dtype=tf.string, name='text')

  text_preprocess_layer =   preprocessing_layer(text_input)
  bert_outputs          =   encoder(text_preprocess_layer)

  net                   =   bert_outputs['pooled_output']
  net                   =   keras.layers.Dropout(0.1)(net)
  net                   =   keras.layers.Dense(17, activation='sigmoid', name='classifier')(net)
  
  return keras.Model(inputs=[text_input], outputs=[net])


def create_history_report(history, epochs, path):
    epochs_stop=np.where(history.history['val_binary_accuracy'] == np.max(history.history['val_binary_accuracy']))
    report = f'''Total_epochs = {epochs}
Best epoch (checkpoint) = {epochs_stop[0][0]}
Final accuracy = {history.history['binary_accuracy'][-1]}
Final val_accuracy = {history.history['val_binary_accuracy'][-1]}
Checkpoint accuracy = {history.history['binary_accuracy'][epochs_stop[0][0]]}
Checkpoint val_accuracy = {history.history['val_binary_accuracy'][epochs_stop[0][0]]}
    '''
    with open(f'{path}/history_report.txt', 'w') as fp:
        fp.write(report)
    # Plot training accuracy
    train_g, = plt.plot(history.history['binary_accuracy'], label='train')
    val_g, = plt.plot(history.history['val_binary_accuracy'], label='val')
    plt.legend(handles=[train_g, val_g], loc='upper left')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.savefig(f'{path}/accuracy_plot.png')


def run_model_test(model_name, model_path, checkpoint:bool):
    model_number = model_name[5:] # modelXX
    run_model = model_execution(model_number, sigmoid=True, threshold=0.5)
    run_model.run_test(save_report=f'{model_path}/{model_name}/test_report.txt', verbose=True)
    if checkpoint:
        run_model = model_execution(model_number, checkpoint=True, sigmoid=True, threshold=0.5)
        run_model.run_test(save_report=f'{model_path}/{model_name}/checkpoint_report.txt', verbose=True)
    
