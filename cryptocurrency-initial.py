import pandas as pd
import numpy as np
from collections import deque
import random
import time
import tensorflow as tf
from sklearn import preprocessing
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
from keras.callbacks import TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt

#constants
SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
RATIO_TO_PREDICT = "BTC-USD"
EPOCHS = 3
BATCH_SIZE = 64
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"

# get data set
def get_data():
    main_df = pd.DataFrame()
    ratios = ["BTC-USD", "BCH-USD", "ETH-USD", "LTC-USD"]

    for ratio in ratios:
        print(ratio)
        dataset = f'{ratio}.csv'
        df = pd.read_csv(dataset, names=['time', 'low', 'high', 'open', 'close', 'volume'])
        df.rename(columns={"close": f"{ratio}_close", "volume": f"{ratio}_volume"}, inplace=True)
        df.set_index("time", inplace=True)
        df = df[[f"{ratio}_close", f"{ratio}_volume"]]

        if len(main_df) == 0:
            main_df = df
        else:
            main_df = main_df.join(df)
    main_df.fillna(method="ffill", inplace=True)
    main_df.dropna(inplace=True)
    print(main_df.head())

    return main_df

def classify(current, future):
  if float(future) > float(current):
    return 1
  else:
    return 0


def preprocess_df(df):
    df = df.drop("future", 1)

    for col in df.columns:
        if col != "target":
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)
    df.dropna(inplace=True)

    sequential_data = []
    prev_data = deque(maxlen=SEQ_LEN)

    for i in df.values:
        prev_data.append([n for n in i[:-1]])
        if len(prev_data) == SEQ_LEN:
            sequential_data.append([np.array(prev_data), i[-1]])
    random.shuffle(sequential_data)

    buys = []
    sells = []

    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])

    random.shuffle(buys)
    random.shuffle(sells)

    lower = min(len(buys), len(sells))

    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys + sells
    random.shuffle(sequential_data)

    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), y

def get_model():
    model = Sequential()
    model.add(LSTM(128, input_shape=(x_train.shape[1:]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())

    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(2, activation='softmax'))

    opt = keras.optimizers.Adam(lr=0.001, decay=1e-6)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data = (x_val, y_val),
                        #callbacks=[tensorboard, checkpoint],
                        )

    return model, history

def plot_results(history):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


main_df = get_data()

main_df['future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)
main_df['target'] = list(map(classify, main_df[f'{RATIO_TO_PREDICT}_close'], main_df['future']))
main_df.dropna(inplace=True)
main_df.head()

times = sorted(main_df.index.values)
last_5pct = sorted(main_df.index.values)[-int(0.05*len(times))]

validation_main_df = main_df[(main_df.index >= last_5pct)]
main_df = main_df[(main_df.index < last_5pct)]

x_train, y_train = preprocess_df(main_df)
x_val, y_val = preprocess_df(validation_main_df)

print(f"train data: {len(x_train)} validation: {len(x_val)}")
print(f"Dont buys: {y_train.count(0)}, buys: {y_train.count(1)}")
print(f"VALIDATION Dont buys: {y_val.count(0)}, buys: {y_val.count(1)}")

model, history = get_model()

score = model.evaluate(x_val, y_val, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

plot_results(history)