
import functools
import numpy as np
from pathlib import Path
import random
import sys
import tensorflow as tf
import tensorflow_addons as tfa
import torch

from masked_conv import masked_conv1d_and_max

if len(sys.argv) == 4:
    DATADIR = '../data/' + '-'.join(sys.argv[1:4])
    RESULTDIR = 'results_' + '-'.join(sys.argv[1:4])
else:
    DATADIR = '../data/example'
    RESULTDIR = 'results'

Path(RESULTDIR).mkdir(exist_ok=True)
Path(RESULTDIR+'/model').mkdir(exist_ok=True)

num_tags = 0
with open(DATADIR + '/vocab.tags.txt', encoding = 'UTF-8') as f:
    for _ in f:
        num_tags += 1


def parse_fn(line_words, line_tags):
    # Encode in Bytes for TF
    words = [w.encode() for w in line_words.strip().split()]
    tags = [t.encode() for t in line_tags.strip().split()]
    if not (len(words) == len(tags)) :
        print(words)
        print(tags )
    assert len(words) == len(tags), "Words and tags lengths don't match"

    # Chars
    chars = [[c.encode() for c in w] for w in line_words.strip().split()]
    lengths = [len(c) for c in chars]
    max_len = max(lengths)
    chars = [c + [b'<pad>'] * (max_len - l) for c, l in zip(chars, lengths)]
    return ((words, len(words)), (chars, lengths)), tags


def generator_fn(words, tags):
    with Path(words).open('r', encoding = 'UTF-8') as f_words, Path(tags).open('r', encoding = 'UTF-8') as f_tags:
        for line_words, line_tags in zip(f_words, f_tags):
            yield parse_fn(line_words, line_tags)


def input_fn(words, tags, params=None, shuffle_and_repeat=False):
    params = params if params is not None else {}
    shapes = ((([None], ()),               # (words, nwords)
               ([None, None], [None])),    # (chars, nchars)
              [None])                      # tags
    types = (((tf.string, tf.int32),
              (tf.string, tf.int32)),
             tf.string)
    defaults = ((('<pad>', 0),
                 ('<pad>', 0)),
                'O')
    dataset = tf.data.Dataset.from_generator(
        functools.partial(generator_fn, words, tags),
        output_shapes=shapes, output_types=types)

    if shuffle_and_repeat:
        dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])

    dataset = (dataset
               .padded_batch(params.get('batch_size', 40), shapes, defaults)
               .prefetch(1))
    return dataset


initializer = tf.keras.initializers.GlorotUniform(random.randint(1, 1024))

class Target_Model(tf.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.vocab_chars = tf.lookup.StaticHashTable(
            tf.lookup.TextFileInitializer(
                filename = DATADIR + '/vocab.chars.txt',
                key_dtype=tf.string, key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
                value_dtype=tf.int64, value_index=tf.lookup.TextFileIndex.LINE_NUMBER
            ),
            default_value = 0
        )
        # embedding matrix for characters, (# of chars + 1) * embedding size
        self.chars_embeddings = tf.Variable(initializer(shape = (41, 50)), dtype = tf.float32)
        self.dropout = tf.keras.layers.Dropout(.5)
        
        self.vocab_words = tf.lookup.StaticHashTable(
            tf.lookup.TextFileInitializer(
                filename = DATADIR + '/vocab.words.txt',
                key_dtype=tf.string, key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
                value_dtype=tf.int64, value_index=tf.lookup.TextFileIndex.LINE_NUMBER
            ),
            default_value = 0
        )
        self.words_embeddings = tf.Variable(
            np.vstack([np.load(str(Path(DATADIR, 'glove.npz')))['embeddings'], [[0.] * 300]]),
            dtype=tf.float32, trainable=False)

        self.lstm_fw = tf.keras.layers.LSTM(100, return_sequences = True)
        self.lstm_bw = tf.keras.layers.LSTM(100, return_sequences = True, go_backwards = True)

        self.num_tags = num_tags
        self.dense = tf.keras.layers.Dense(self.num_tags)

        self.crf = tfa.layers.CRF(self.num_tags)

        self.canary = tf.constant(0)

    def __call__(self, element, training):
        # settings
        (words, nwords), (chars, nchars) = element
        # Char Embeddings
        char_ids = self.vocab_chars.lookup(chars)
        char_embeddings = tf.nn.embedding_lookup(self.chars_embeddings, char_ids)
        char_embeddings = self.dropout(char_embeddings, training=training)
        # Char 1d convolution
        weights = tf.sequence_mask(nchars)
        char_embeddings = masked_conv1d_and_max(
            char_embeddings, weights, 50, 3)
        # Word Embeddings
        word_ids = self.vocab_words.lookup(words)
        word_embeddings = tf.nn.embedding_lookup(self.words_embeddings, word_ids)
        # Concatenate Word and Char Embeddings
        embeddings = tf.concat([word_embeddings, char_embeddings], axis=-1)
        embeddings =  self.dropout(embeddings, training=training)
        # bi-LSTM
        output_fw = self.lstm_fw(embeddings)
        output_bw = self.lstm_bw(embeddings)
        output = tf.concat([output_fw, output_bw], axis=-1)
        # dropout, dense, activation
        output = self.dropout(output, training=training)
        output = self.dense(output)

        # CRF layer
        return self.crf(output)

    def canary_extraction(self, pattern, n):
        #if pattern == 'call' :
        token_set = tf.constant(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
        char_ids = self.vocab_chars.lookup(token_set)
        word_ids = self.vocab_words.lookup(token_set)
        
        char_embedding_matrix = tf.nn.embedding_lookup(self.chars_embeddings, char_ids)
        word_embedding_matrix = tf.nn.embedding_lookup(self.words_embeddings, word_ids)
        
        # learnable input (logit vector)
        target_softmax = tf.nn.softmax(self.canary)

        # Chars Embeddings
        # construct char embedding vector of target (learnable part)
        target_char = tf.tensordot(target_softmax, char_embedding_matrix, 1)
        target_char = tf.expand_dims(target_char, axis=1)
        pad_char = self.vocab_chars.lookup(tf.constant(['<pad>'] * 3))
        pad_char = tf.nn.embedding_lookup(self.chars_embeddings, pad_char)
        pad_char = tf.expand_dims(pad_char, axis=0)
        pad_char = tf.repeat(pad_char, repeats = n, axis = 0)
        target_char = tf.concat([target_char, pad_char], 1)
        # construct char embedding vector of prefix
        if pattern == 'call' :
            prefix_char = self.vocab_chars.lookup(tf.constant([['c', 'a', 'l', 'l']]))
        elif pattern == 'pin' :
            prefix_char = self.vocab_chars.lookup(tf.constant(
                [["m", "y", '<pad>', '<pad>'], ["p", "i", "n", '<pad>'], ["c", "o", "d", "e"], ["i", "s", '<pad>', '<pad>']]))
        prefix_char = tf.nn.embedding_lookup(self.chars_embeddings, prefix_char)
        # concatenate
        char_embeddings = tf.concat([prefix_char, target_char], 0)
        # Char 1d convolution
        if pattern == 'call':
            weights = tf.sequence_mask([4] + [1] * n)
        elif pattern == 'pin':
            weights = tf.sequence_mask([2, 3, 4, 2] + [1] * n)
        char_embeddings = masked_conv1d_and_max(
            char_embeddings, weights, 50, 3)
        
        # Word Embeddings
        target_word = tf.tensordot(target_softmax, word_embedding_matrix, 1)
        if pattern == 'call':
            prefix_word = self.vocab_words.lookup(tf.constant(['call']))
        elif pattern == 'pin':
            prefix_word = self.vocab_words.lookup(tf.constant(['my', 'pin', 'code', 'is']))
        prefix_word = tf.nn.embedding_lookup(self.words_embeddings, prefix_word)
        word_embeddings = tf.concat([prefix_word, target_word], 0)

        # Concatenate Word and Char Embeddings
        embeddings = tf.concat([word_embeddings, char_embeddings], axis=-1)
        embeddings = tf.expand_dims(embeddings, axis=0)

        # bi-LSTM & dense
        output_fw = self.lstm_fw(embeddings)
        output_bw = self.lstm_bw(embeddings)
        output = tf.concat([output_fw, output_bw], axis=-1)
        output = self.dense(output)
        
        # CRF layer
        return self.crf(output)


params = {
    'dim_chars': 100,
    'dim': 300,
    'dropout': 0.5,
    'num_oov_buckets': 1,
    'epochs': 1,
    'batch_size': 64,
    'buffer': 15000,
    'filters': 50,
    'kernel_size': 3,
    'lstm_size': 100,
    'words': str(Path(DATADIR, 'vocab.words.txt')),
    'chars': str(Path(DATADIR, 'vocab.chars.txt')),
    'tags': str(Path(DATADIR, 'vocab.tags.txt')),
    'glove': str(Path(DATADIR, 'glove.npz'))
}



vocab_tags = tf.lookup.StaticHashTable(
    tf.lookup.TextFileInitializer(
        filename = DATADIR + '/vocab.tags.txt',
        key_dtype=tf.string, key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
        value_dtype=tf.int64, value_index=tf.lookup.TextFileIndex.LINE_NUMBER
    ),
    default_value = num_tags - 1
)

vocab_tags_decode = tf.lookup.StaticHashTable(
    tf.lookup.TextFileInitializer(
        filename = DATADIR + '/vocab.tags.txt',
        key_dtype=tf.int64, key_index=tf.lookup.TextFileIndex.LINE_NUMBER,
        value_dtype=tf.string, value_index=tf.lookup.TextFileIndex.WHOLE_LINE
    ),
    default_value = 'O'
)


train_input = input_fn(Path(DATADIR, 'train.words.txt'), Path(DATADIR, 'train.tags.txt'),
                                params, shuffle_and_repeat=True)
test_a = input_fn(Path(DATADIR, 'testa.words.txt'), Path(DATADIR, 'testa.tags.txt'),
                                params, shuffle_and_repeat=True)
test_b = input_fn(Path(DATADIR, 'testb.words.txt'), Path(DATADIR, 'testb.tags.txt'),
                                params, shuffle_and_repeat=True)

model = Target_Model()
checkpoint = tf.train.Checkpoint(model)


optimizer = tf.keras.optimizers.Adam(0.02)
train_loss = tf.keras.metrics.Mean(name="train_loss")
valid_loss = tf.keras.metrics.Mean(name="valid_loss")

def loss_func(potentials, sequence_length, kernel, y):
    crf_likelihood, _ = tfa.text.crf_log_likelihood(
        potentials, y, sequence_length, kernel
    )
    return tf.reduce_mean(-1 * crf_likelihood)


def train(x, y):
    with tf.GradientTape() as t:
        decoded_sequence, potentials, sequence_length, kernel = model(x, True)
        loss = loss_func(potentials, sequence_length, kernel, y)
    grads = t.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss(loss)

def validate(data):
    valid_loss.reset_states()
    correct = 0
    total = 0
    for x, y in data:
        (words, nwords), (chars, nchars) = x
        decoded_sequence, potentials, sequence_length, kernel = model(x, False)
        y = vocab_tags.lookup(y)
        for (y_pred, y_real, length) in zip(decoded_sequence.numpy(), y.numpy(), nwords) :
            for i in range(length) :
                if y_pred[i] == y_real[i] :
                    correct += 1
            total += length
        loss = loss_func(potentials, sequence_length, kernel, y)
        valid_loss(loss)
    return (valid_loss.result(), correct / total)

train_losses = []
valid_losses = []
checkpoints = []



for epoch in range(20):
    train_loss.reset_states()
    for x, y in train_input:
        y = vocab_tags.lookup(y)
        train(x, y)
    loss, acc = validate(test_a)
    
    save_path = checkpoint.save(RESULTDIR+'/model/checkpoint')
    checkpoints.append(save_path)

    train_losses.append(train_loss.result())
    valid_losses.append(loss)

    print(f"Epoch {epoch + 1}, Loss: {train_loss.result()}")
    print(f"Validation, Loss: {loss}, Accuracy: {acc}")
    
loss, acc = validate(test_b)
print(f"Test, Loss: {loss}, Accuracy: {acc}")

def write_predictions(name):
    Path(RESULTDIR + '/score').mkdir(parents=True, exist_ok=True)
    with Path(RESULTDIR + '/score/{}.preds.txt'.format(name)).open('wb') as f:
        test_inpf = input_fn(Path(DATADIR, name + '.words.txt'), Path(DATADIR, name + '.tags.txt'), params)
        for x, tags in test_inpf:
            (words, _), _ = x
            decoded_sequence, _, _, kernel = model(x, False)
            preds = vocab_tags_decode.lookup(tf.cast(decoded_sequence, tf.int64))
            for words_, tags_, preds_ in zip(words, tags, preds):
                for word, tag, tag_pred in zip(words_, tags_, preds_):
                    if word.numpy() != b'<pad>' :
                        f.write(b' '.join([word.numpy(), tag.numpy(), tag_pred.numpy()]) + b'\n')
                f.write(b'\n')

for name in ['train', 'testa', 'testb']:
    write_predictions(name)

if len(sys.argv) == 1:
    exit()

########################


def canary_extraction(pattern, n):
    optimizer = tf.keras.optimizers.Adam(0.02)
    if pattern == 'call':
        y = [['O', 'B-canary'] + ['I-canary'] * (n - 1)]
    elif pattern == 'pin':
        y = [['O', 'O', 'O', 'O', 'B-canary'] + ['I-canary'] * (n - 1)]
    y = vocab_tags.lookup(tf.constant(y))
    model.canary = tf.Variable(initializer(shape = [n, 10]), dtype = tf.float32)
    c = [-1] * n
    same = 0
    for i in range(1000):
        with tf.GradientTape() as t:
            decoded_sequence, potentials, sequence_length, kernel = model.canary_extraction(pattern, n)
            loss = loss_func(potentials, sequence_length, kernel, y)
        grad = t.gradient(loss, [model.canary])
        optimizer.apply_gradients(zip(grad, [model.canary]))
        temp = tf.math.argmax(model.canary, -1).numpy()
        if not (c == temp).all() :
            print(i, temp)
            c = temp
            same = 0
        else:
            same += 1
        if same == 100 :
            print(i)
            break
    with Path(RESULTDIR + '/score/canary.txt').open('a', encoding = 'UTF-8') as f:
        f.write(str(tf.math.argmax(model.canary, -1).numpy())+'\n')


canary_extraction(sys.argv[1], int(sys.argv[2]))

####################