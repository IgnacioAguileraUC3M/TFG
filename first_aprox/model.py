import tensorflow as tf
import numpy as np
import pandas as pd

SEED = 1 # Para los datasets
LEARNING_RATE = 0.01

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
#----------------------------------------------------------
#--------------------Decode labels-------------------------
#----------------------------------------------------------
for i, label in enumerate(train_dataset.class_names):
  print("Label", i, "corresponds to", label)
# exit()
#----------------------------------------------------------
#----------------------------------------------------------
#----------------------------------------------------------
# exit()
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


history = model.fit(train_dataset, epochs=10,
                    validation_data=validation_dataset,
                    validation_steps=1)


model.save('./first_aprox/models/model_1.tf', save_format='tf')