from keras.datasets import imdb
import numpy as np

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10_000)


def vectorize_sequences(sequences, dimension=10_000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation = 'relu', input_shape = (10_000,)))
model.add(layers.Dense(16, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train,
            y_train,
            epochs=4,
            batch_size=512)

results = model.evaluate(x_test, y_test)

from test import encode
from review_prepare import prepare

# %%
while True:
    review = input("Type \"q\" to exit.\nWrite or paste a film review: ")
    if review == "q":
        break

    review_enc = encode(prepare(review))
    review_arr = np.empty(1, dtype='O')
    review_arr = review_enc,
    review_vec = vectorize_sequences(review_arr)

    res = model(review_vec)

    print(f"The probability of the review being a positive one is {round(res.numpy()[0][0]*100, 0)}%")
    print()