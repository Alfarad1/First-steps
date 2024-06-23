from keras.datasets import imdb

word_index = imdb.get_word_index()
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()])

def decode(review):
    return ' '.join([reverse_word_index.get(i-3, '?') for i in review])

def encode(review: str):
    result = list()
    for i in review.split(' '):
        if i in word_index.keys() and word_index[i] + 3 <= 10000: # ограничиваем словарь 10 000 словами
            result.append(word_index[i] + 3)
        else:
            result.append(1)
    return result



if __name__ == "__main__":
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10_000)

    while True:
        review = train_data[int(input("review number: "))]
        decoded = decode(review)
        print (decoded)
        # encoded = encode(decoded)