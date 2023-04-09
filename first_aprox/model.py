import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import class_weight

SEED = 1 # Para los datasets
LEARNING_RATE = 0.0001

train_dataset = tf.keras.utils.text_dataset_from_directory('./first_aprox/data/dataset', 
                                                           seed=SEED,
                                                           labels = 'inferred', 
                                                           label_mode='categorical',
                                                           subset = 'training',
                                                           validation_split=0.2)

validation_dataset = tf.keras.utils.text_dataset_from_directory('./first_aprox/data/dataset', 
                                                           seed=SEED,
                                                           labels = 'inferred', 
                                                           label_mode='categorical',
                                                           subset = 'validation',
                                                           validation_split=0.2)

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

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                    loss='categorical_crossentropy',
                    metrics=['categorical_accuracy'])

class_weight_dict = dict(enumerate(class_weights))
history = model.fit(train_dataset, epochs=27,
                    validation_data=validation_dataset,
                    validation_steps=1, 
                    class_weight=class_weight_dict)


model.save('./first_aprox/models/model_2.h5', save_format='h5')