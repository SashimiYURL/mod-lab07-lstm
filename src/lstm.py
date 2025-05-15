import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
from keras.optimizers import Adam
import random
import os

with open('src/input.txt', 'r', encoding='utf-8') as file:
    text = file.read().lower()

words = text.split()
vocab = sorted(list(set(words)))
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for i, word in enumerate(vocab)}

seq_length = 10
vocab_size = len(vocab)

sequences = []
next_words = []
for i in range(0, len(words) - seq_length):
    sequences.append(words[i:i + seq_length])
    next_words.append(words[i + seq_length])


X = np.zeros((len(sequences), seq_length, vocab_size), dtype=np.bool)
y = np.zeros((len(sequences), vocab_size), dtype=np.bool)

for i, seq in enumerate(sequences):
    for t, word in enumerate(seq):
        X[i, t, word_to_idx[word]] = 1
    y[i, word_to_idx[next_words[i]]] = 1


model = Sequential()
model.add(LSTM(128, input_shape=(seq_length, vocab_size)))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
             optimizer=Adam(learning_rate=0.01),
             metrics=['accuracy'])

model.fit(X, y, epochs=50, batch_size=128)

def generate_text(word_count, model, words, word_to_idx, idx_to_word, seq_length):
    start_idx = random.randint(0, len(words) - seq_length - 1)
    current_words = words[start_idx:start_idx + seq_length]
    
    generated = []
    for _ in range(word_count):
        x = np.zeros((1, seq_length, vocab_size))
        for t, word in enumerate(current_words):
            x[0, t, word_to_idx[word]] = 1.

        preds = model.predict(x, verbose=0)[0]
        
        temperature = 0.7
        preds = np.log(preds + 1e-10) / temperature  
        exp_preds = np.exp(preds - np.max(preds))    
        preds = exp_preds / np.sum(exp_preds)
        
        if not np.all(np.isfinite(preds)) or np.sum(preds) <= 0:
            preds = np.ones_like(preds)/len(preds)  
        
        try:
            next_idx = np.random.choice(len(preds), p=preds)
        except ValueError:
            next_idx = np.argmax(preds)
            
        next_word = idx_to_word[next_idx]
        generated.append(next_word)
        current_words = current_words[1:] + [next_word]
    
    formatted_text = []
    for i in range(0, len(generated), 30):
        formatted_text.append(' '.join(generated[i:i+30]))
    return '\n'.join(formatted_text)


generated_text = generate_text(
    word_count=1000,
    model=model,
    words=words,
    word_to_idx=word_to_idx,
    idx_to_word=idx_to_word,
    seq_length=seq_length
)


with open('src/gen.txt', 'w', encoding='utf-8') as file:
    file.write(generated_text)

print("Генерация завершена. Результат сохранен в result/gen.txt")
print("Первые 100 слов:")
print('\n'.join(generated_text.split('\n')[:4])) 