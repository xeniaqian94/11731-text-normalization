from random import randint
import sys
import theano
import numpy as np
import theano.tensor as T
import lasagne
import time
import cPickle
from crf import CRFLayer
from objectives import crf_loss, crf_accuracy

MAX_CHAR_LENGTH = 25
MAX_WORD_LENGTH = 41

NUM_EPOCH = 50

char_embed_size = 50
char_hidden_size = 256
word_hidden_size = 512


def create_char_index():
    char_dict = []
    with open('./data/train_data_multiline_CAND', 'r') as file:
        for line in file:
            l = line.rstrip()
            if len(l) != 0:
                word = l.split(' ')[0]
                for i in range(len(word)):
                    if word[i] not in char_dict:
                        char_dict.append(word[i])
    with open('./data/valid_data_multiline_CAND', 'r') as file:
        for line in file:
            l = line.rstrip()
            if len(l) != 0:
                word = l.split(' ')[0]
                for i in range(len(word)):
                    if word[i] not in char_dict:
                        char_dict.append(word[i])
    with open('./data/test_truth_multiline_CAND', 'r') as file:
        for line in file:
            l = line.rstrip()
            if len(l) != 0:
                word = l.split(' ')[0]
                for i in range(len(word)):
                    if word[i] not in char_dict:
                        char_dict.append(word[i])
    print len(char_dict)

    return char_dict

def read_sentence(word_dict, word_vector, char_dict, p):

    char_input = np.zeros((2000, MAX_WORD_LENGTH, MAX_CHAR_LENGTH), dtype = 'int32')
    word_input = np.zeros((2000, MAX_WORD_LENGTH, 100), dtype = 'float32')
    char_mask = np.zeros((2000, MAX_WORD_LENGTH, MAX_CHAR_LENGTH), dtype = 'float32')
    word_mask = np.zeros((2000, MAX_WORD_LENGTH), dtype = 'float32')
    labels = np.zeros((2000, MAX_WORD_LENGTH), dtype = 'int32')

    # sentences = []
    # sentence = []
    # labels = []
    # label = []

    i = 0
    j = 0

    with open('./data/' + p + '_data_multiline_CAND', 'r') as file:
        for line in file:
            l = line.rstrip()
            if len(l) != 0:
                word = l.split(' ')[0]
                label = 0
                if l.split(' ')[6] == 'CAND':
                    label = 1

                if word in word_dict:
                    word_input[i][j] = word_vector[word_dict.index(word)]

                for k in range(len(word)):
                    if k < MAX_CHAR_LENGTH:
                        char_input[i][j][k] = char_dict.index(word[k])

                labels[i][j] = label

                char_mask[i][j][:len(word)] = 1
                word_mask[i][j] = 1

                j += 1

            else:

                i += 1
                j = 0

    return char_input, word_input, char_mask, word_mask, labels

def read_glove():
    word = []
    vector = []
    with open('glove.twitter.27B.100d.txt', 'r') as file:
        for line in file:
            l = line.rstrip().split(' ')
            word.append(l[0])
            vec = np.array([float(i) for i in l[1:]], dtype = 'float32')
            vector.append(vec)
    return word, vector

def iterate_minibatches(char_inputs, char_masks, word_inputs, word_masks, targets, batch_size=10, shuffle=False):
    assert len(word_inputs) == len(targets)
    assert len(word_inputs) == len(word_masks)
    assert len(word_inputs) == len(char_inputs)
    assert len(word_inputs) == len(char_masks)
    if shuffle:
        indices = np.arange(len(word_inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(word_inputs), batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield char_inputs[excerpt], char_masks[excerpt], word_inputs[excerpt], word_masks[excerpt], targets[excerpt]
            
def main():

    print 'building character index...'
    char_dict = create_char_index()
    print 'done'

    print 'reading glove word vectors...'
    word_dict, word_vector = read_glove()
    print 'done'

    print 'reading training data...'
    train_char_X, train_word_X, train_char_mask, train_word_mask, train_Y = read_sentence(word_dict, word_vector, char_dict, 'train')
    print 'done'

    print 'reading validation data...'
    valid_char_X, valid_word_X, valid_char_mask, valid_word_mask, valid_Y = read_sentence(word_dict, word_vector, char_dict, 'valid')
    print 'done'

    print 'building model...'

    char_input_var = T.itensor3()
    char_mask_var = T.tensor3(dtype = 'float32')
    target_var = T.imatrix()
    mask_var = T.matrix(dtype = 'float32')

    num_tokens = mask_var.sum(dtype=theano.config.floatX)

    char_in = lasagne.layers.InputLayer(shape=(None, MAX_WORD_LENGTH, MAX_CHAR_LENGTH), input_var = char_input_var)
    char_embed = lasagne.layers.EmbeddingLayer(char_in, len(char_dict), char_embed_size)
    char_embed = lasagne.layers.ReshapeLayer(char_embed, (-1, [2], [3]))

    char_mask_in = lasagne.layers.InputLayer(shape=(None, MAX_WORD_LENGTH, MAX_CHAR_LENGTH), input_var = char_mask_var)
    char_mask = lasagne.layers.ReshapeLayer(char_mask_in, (-1, [2]))

    char_forward = lasagne.layers.LSTMLayer(char_embed, char_hidden_size, mask_input = char_mask, nonlinearity=lasagne.nonlinearities.tanh)
    char_forward = lasagne.layers.ReshapeLayer(char_forward, (-1, MAX_WORD_LENGTH, [1], [2]))
    char_forward = lasagne.layers.SliceLayer(char_forward, -1, axis = 2)

    char_backward = lasagne.layers.LSTMLayer(char_embed, char_hidden_size, mask_input = char_mask, nonlinearity=lasagne.nonlinearities.tanh, backwards=True)
    char_backward = lasagne.layers.ReshapeLayer(char_backward, (-1, MAX_WORD_LENGTH, [1], [2]))
    char_backward = lasagne.layers.SliceLayer(char_backward, 0, axis = 2)

    char_concat = lasagne.layers.ConcatLayer([char_forward, char_backward], axis = 2)

    word_in = lasagne.layers.InputLayer(shape=(None, MAX_WORD_LENGTH, 100))
    word_concat = lasagne.layers.ConcatLayer([char_concat, word_in], axis = 2)
    word_mask = lasagne.layers.InputLayer(shape=(None, MAX_WORD_LENGTH), input_var = mask_var)

    word_forward = lasagne.layers.LSTMLayer(word_concat, word_hidden_size, mask_input = word_mask, nonlinearity = lasagne.nonlinearities.tanh)
    word_backward = lasagne.layers.LSTMLayer(word_concat, word_hidden_size, mask_input = word_mask, nonlinearity = lasagne.nonlinearities.tanh, backwards=True)
    word_c = lasagne.layers.ConcatLayer([word_forward, word_backward], axis = 2)

    crf = CRFLayer(word_c, 2, mask_input = word_mask)

    energies_train = lasagne.layers.get_output(crf)
    loss = crf_loss(energies_train, target_var, mask_var).mean()

    all_params = lasagne.layers.get_all_params(crf, trainable = True)
    updates = lasagne.updates.sgd(loss, all_params, learning_rate = 0.1)

    energies_eval = lasagne.layers.get_output(crf, deterministic=True)
    loss_eval = crf_loss(energies_eval, target_var, mask_var).mean()

    _, corr_train = crf_accuracy(energies_train, target_var)
    corr_train = (corr_train * mask_var).sum(dtype=theano.config.floatX)
    prediction_eval, corr_eval = crf_accuracy(energies_eval, target_var)
    corr_eval = (corr_eval * mask_var).sum(dtype=theano.config.floatX)

    train = theano.function([char_input_var, char_mask_var, word_in.input_var, mask_var, target_var], [loss, corr_train, num_tokens], updates = updates)

    valid_eval = theano.function([char_input_var, char_mask_var, word_in.input_var, mask_var, target_var], [loss_eval, corr_eval, prediction_eval, num_tokens])

    print 'done'

    print 'training...'

    for epoch in range(1, NUM_EPOCH + 1):
        
        loss = 0
        train_corr = 0
        train_batches = 0
        train_total = 0
        start_time = time.time()
        
        for batch in iterate_minibatches(train_char_X, train_char_mask, train_word_X, train_word_mask, train_Y):
            c_input, c_mask, w_input, w_mask, y = batch
            batchloss, corr, num = train(c_input, c_mask, w_input, w_mask, y)
            loss += batchloss
            train_corr += corr
            train_total += num
            train_batches += 1

        print("Epoch {} of {} took {:.3f}s".format(epoch, NUM_EPOCH, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(loss/train_batches))
        print("  training accuracy:\t\t{:.6f}".format(train_corr * 100.0 / train_total))

        valid_corr = 0
        valid_total = 0
        if epoch % 5 == 0:
            for batch in iterate_minibatches(valid_char_X, valid_char_mask, valid_word_X, valid_word_mask, valid_Y):
                c_input, c_mask, w_input, w_mask, y = batch
                loss, corr, pred, num = valid_eval(c_input, c_mask, w_input, w_mask, y)
                print pred
                valid_corr += corr
                valid_total += num
            print("  valid accuracy:\t\t{:.6f}".format(valid_corr * 100.0 / valid_total))

    

if __name__ == '__main__':
    main()
