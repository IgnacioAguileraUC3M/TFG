import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from sklearn.utils import class_weight
from src import new_model_name
from src import model_execution
from matplotlib import pyplot as plt
import os
import random
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
# import shutil
# try:
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer
# except ModuleNotFoundError:
#     pass
# try:
# except:
    # pass

def load_data(ds_path:str, batch_size:int, seed:int = None, shuffle:bool = False):
    # TF_GPU_ALLOCATOR=cuda_malloc_async
    if seed is None:
            print('Random seed')
            seed = random.randint(1,1000)

    train_dataset = keras.utils.text_dataset_from_directory(ds_path, 
                                                            seed=seed,
                                                            labels = 'inferred', 
                                                            label_mode='categorical',
                                                            subset = 'training',
                                                            validation_split=0.2,
                                                            shuffle=shuffle,
                                                            batch_size=batch_size)

    validation_dataset = keras.utils.text_dataset_from_directory(ds_path, 
                                                            seed=seed,
                                                            labels = 'inferred', 
                                                            label_mode='categorical',
                                                            subset = 'validation',
                                                            validation_split=0.2,
                                                            shuffle=shuffle,
                                                            batch_size=batch_size)
    return train_dataset, validation_dataset

def new_model(seed:int = None, 
              lr:float = 0.0001, 
              epochs:int = 10,
              ds_path:str = './v1/data/dataset', 
              model_path:str = './v1/model/models', 
              model_name:str = new_model_name(),
              checkpoint_path:str = f'./v1/model/models/{new_model_name()}/checkpoint.tf', 
              class_weights:bool = False, 
              architecture:str = 'recurrent',
              checkpoint:bool = True,
              shuffle:bool = False,
              batch_size:int=32):

    os.mkdir(f'{model_path}/{model_name}/')
    if seed is None:
        print('Random seed')
        seed = random.randint(1,1000)
    csv_ds = ds_path[-3:] == 'csv'
    if csv_ds: # xxx.csv
        train_data = pd.read_csv(ds_path)
        y_train_data = train_data.iloc[:,-1:]  # Las clases de las instancias
        x_train_data = train_data.iloc[:,:-1]  # Los atributos de las instancias

        encoder = MultiLabelBinarizer()

        y_train_transformed = encoder.fit_transform(y_train_data)


        input_shape = (x_train_data.shape[1],)
        num_classes = y_train_transformed.shape[1]


        x_train, x_val, y_train, y_val = train_test_split(x_train_data, y_train_transformed,
                                                    stratify = y_train_transformed,
                                                    test_size = 0.2)


    else:
        train_dataset, validation_dataset = load_data(ds_path, batch_size, seed, shuffle)

        y_train = np.concatenate([y for x, y in train_dataset], axis=0)

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                                    monitor='val_accuracy',
                                                                    mode='max',
                                                                    save_best_only=True)


    classes_list = []
    y_ints = [y.argmax() for y in y_train]

    class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                    classes=np.unique(y_ints),
                                                    y=y_ints)
    # a = ''
    # for aa in y_train.tolist():
    #     a += str(aa) + '\n'
    # with open('all_t.txt', 'w') as fp:
    #     fp.write(a)
    # exit()
    class_weight_dict = dict(enumerate(class_weights))


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
        keras.layers.Dense(17, activation=keras.activations.softmax)
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                        loss='categorical_crossentropy',
                        metrics=['categorical_accuracy'])

    class_weight_dict = dict(enumerate(class_weights))

    history = model.fit(train_dataset,
                        epochs=epochs,
                        validation_data=validation_dataset,
                        validation_steps=1, 
                        class_weight=class_weight_dict,
                        callbacks=[model_checkpoint_callback],
                        shuffle=shuffle,
                        batch_size=batch_size)


    model.save(f'{model_path}/{model_name}/{model_name}.tf', save_format='tf')
    run_model_test(model_name, model_path)
    create_history_report(history, epochs, f'{model_path}/{model_name}')



def new_bert_model(ds_path:str='./v1/data/dataset', 
                   seed:int=None, 
                   shuffle:bool=False,
                   model_name:str = new_model_name(),
                   model_path:str = './v1/model/models', 
                   checkpoint_path:str = f'./v1/model/models/{new_model_name()}/checkpoint.tf',
                   epochs:int = 5,
                   batch_size:int=5):

    import tensorflow_hub as hub
    import tensorflow_text as text
    from official.nlp import optimization  # to create AdamW optimizer



    csv_ds = ds_path[-3:] == 'csv'
    if csv_ds: # xxx.csv
        train_data = pd.read_csv(ds_path, converters={'CLASS': pd.eval})
        # print(train_data.dtypes)
        # y_train_data = train_data['CLASS'] # Las clases de las instancias
        # x_train_data = train_data['TEXT']  # Los atributos de las instancias
        # print(x_train_data)
        # print(y_train_data)
        encoder = MultiLabelBinarizer()

        # y_train_transformed = encoder.fit_transform(y_train_data)
        # print(y_train_transformed)
        # exit()

        # input_shape = (x_train_data.shape[1],)
        # num_classes = y_train_transformed.shape[1]


        # x_train, x_val, y_train, y_val = train_test_split(x_train_data, y_train_transformed,
        #                                             stratify = y_train_transformed.values,
        #                                             test_size = 0.2,
        #                                             shuffle=shuffle)
        train_df, val_df = train_test_split(train_data,
                                            # stratify = train_data['CLASS'].values,
                                            test_size = 0.2,
                                            shuffle=shuffle)
        x_train = train_df['TEXT']
        x_val = val_df['TEXT']
        y_train_raw = train_df['CLASS']
        y_val_raw = val_df['CLASS']
        y_train = encoder.fit_transform(y_train_raw)
        y_val = encoder.fit_transform(y_val_raw)
        steps_per_epoch = len(x_train)

    else:
        train_dataset, validation_dataset = load_data(ds_path, batch_size, seed, shuffle)
        steps_per_epoch = tf.data.experimental.cardinality(train_dataset).numpy()

    model = build_classifier_model()

    loss = keras.losses.BinaryCrossentropy()
    metrics = tf.metrics.BinaryAccuracy()

    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1*num_train_steps)

    init_lr = 1e-5
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')
    
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)
    if csv_ds:
        history = model.fit(x_train,
                            y_train,
                            validation_data=(x_val, y_val),
                            epochs=epochs,
                            batch_size=batch_size)
    else:
        history = model.fit(train_dataset,
                            validation_data=validation_dataset,
                            epochs=epochs,
                            batch_size=batch_size)

    model.save(f'{model_path}/{model_name}/{model_name}.tf', save_format='tf', include_optimizer=False)
    run_model_test(model_name, model_path, chekpoint=False)
    create_history_report(history, epochs, f'{model_path}/{model_name}')



def build_classifier_model():
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


def run_model_test(model_name, model_path, chekpoint:bool=True):
    model_number = model_name[5:] # modelXX
    run_model = model_execution(model_number, sigmoid=True, threshold=0.5)
    run_model.run_test(save_report=f'{model_path}/{model_name}/test_report.txt', verbose=True)
    if chekpoint:
        run_model = model_execution(model_number, checkpoint=False)
        run_model.run_test(save_report=f'{model_path}/{model_name}/checkpoint_report.txt', verbose=True)
    
