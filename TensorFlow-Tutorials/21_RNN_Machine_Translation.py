import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
import os

from ._conv import register_converters as _register_converters

# from tf.keras.models import Model  # This does not work!
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

tf.__version__

tf.keras.__version__

import europarl

language_code='da'

mark_start = 'ssss '
mark_end = ' eeee'

# data_dir = "data/europarl/"

europarl.maybe_download_and_extract(language_code=language_code)

data_src = europarl.load_data(english=False,
                              language_code=language_code)

data_dest = europarl.load_data(english=True,
                               language_code=language_code,
                               start=mark_start,
                               end=mark_end)

idx = 2

data_src[idx]

data_dest[idx]

idx = 8002

data_src[idx]

data_dest[idx]

num_words = 10000

class TokenizerWrap(Tokenizer):
    """Wrap the Tokenizer-class from Keras with more functionality."""

    def __init__(self, texts, padding,
                 reverse=False, num_words=None):
        """
        :param texts: List of strings. This is the data-set.
        :param padding: Either 'post' or 'pre' padding.
        :param reverse: Boolean whether to reverse token-lists.
        :param num_words: Max number of words to use.
        """
​
        Tokenizer.__init__(self, num_words=num_words)
​
        # Create the vocabulary from the texts.
        self.fit_on_texts(texts)
​
        # Create inverse lookup from integer-tokens to words.
        self.index_to_word = dict(zip(self.word_index.values(),
                                      self.word_index.keys()))
​
        # Convert all texts to lists of integer-tokens.
        # Note that the sequences may have different lengths.
        self.tokens = self.texts_to_sequences(texts)
​
        if reverse:
            # Reverse the token-sequences.
            self.tokens = [list(reversed(x)) for x in self.tokens]

            # Sequences that are too long should now be truncated
            # at the beginning, which corresponds to the end of
            # the original sequences.
            truncating = 'pre'
        else:
            # Sequences that are too long should be truncated
            # at the end.
            truncating = 'post'
​
        # The number of integer-tokens in each sequence.
        self.num_tokens = [len(x) for x in self.tokens]
​
        # Max number of tokens to use in all sequences.
        # We will pad / truncate all sequences to this length.
        # This is a compromise so we save a lot of memory and
        # only have to truncate maybe 5% of all the sequences.
        self.max_tokens = np.mean(self.num_tokens) \
                          + 2 * np.std(self.num_tokens)
        self.max_tokens = int(self.max_tokens)
​
        # Pad / truncate all token-sequences to the given length.
        # This creates a 2-dim numpy matrix that is easier to use.
        self.tokens_padded = pad_sequences(self.tokens,
                                           maxlen=self.max_tokens,
                                           padding= padding,
                                           truncating= truncating)
​
    def token_to_word(self, token):
        """Lookup a single word from an integer-token."""
​
        word = " " if token == 0 else self.index_to_word[token]
        return word
​
    def tokens_to_string(self, tokens):
        """Convert a list of integer-tokens to a string."""
​
        # Create a list of the individual words.
        words = [self.index_to_word[token]
                 for token in tokens
                 if token != 0]

        # Concatenate the words to a single string
        # with space between all the words.
        text = " ".join(words)
​
        return text

    def text_to_tokens(self, text, reverse=False, padding=False):
        """
        Convert a single text-string to tokens with optional
        reversal and padding.
        """
​
        # Convert to tokens. Note that we assume there is only
        # a single text-string so we wrap it in a list.
        tokens = self.texts_to_sequences([text])
        tokens = np.array(tokens)
​
        if reverse:
            # Reverse the tokens.
            tokens = np.flip(tokens, axis=1)
​
            # Sequences that are too long should now be truncated
            # at the beginning, which corresponds to the end of
            # the original sequences.
            truncating = 'pre'
        else:
            # Sequences that are too long should be truncated
            # at the end.
            truncating = 'post'
​
        if padding:
            # Pad and truncate sequences to the given length.
            tokens = pad_sequences(tokens,
                                   maxlen=self.max_tokens,
                                   padding='pre',
                                   truncating=truncating)
​
        return tokens

tokenizer_src = TokenizerWrap(texts=data_src,
                              padding='pre',
                              reverse=True,
                              num_words=num_words)


tokenizer_dest = TokenizerWrap(texts=data_dest,
                               padding = 'post',
                               reverse = False,
                               num_words=num_words)

tokens_src = tokenizer_src.tokens_padded
tokens_dest = tokenizer_dest.tokens_padded
print(tokens_src.shape)
print(tokens_dest.shape)

token_start = tokenizer_dest.word_index[mark_start.strip()]
token_start

token_end = tokenizer_dest.word_index[mark_end.strip()]
token_end

idx = 2

tokens_src[idx]

tokenizer_src.tokens_to_string(tokens_src[idx])

data_src[idx]

tokens_dest[idx]

tokenizer_dest.tokens_to_string(tokens_dest[idx])

data_dest[idx]

encoder_input_data = tokens_src

decoder_input_data = tokens_dest[:, :-1]
decoder_input_data.shape

decoder_output_data = tokens_dest[:, 1:]
decoder_output_data.shape

idx = 2

decoder_input_data[idx]

decoder_output_data[idx]

tokenizer_dest.tokens_to_string(decoder_input_data[idx])

tokenizer_dest.tokens_to_string(decoder_output_data[idx])

encoder_input = Input(shape=(None, ), name='encoder_input')

embedding_size = 128

encoder_embedding = Embedding(input_dim=num_words,
                              output_dim=embedding_size,
                              name='encoder_embedding')

state_size = 512

encoder_gru1 = GRU(state_size, name='encoder_gru1',
                   return_sequences=True)
encoder_gru2 = GRU(state_size, name='encoder_gru2',
                   return_sequences=True)
encoder_gru3 = GRU(state_size, name='encoder_gru3',
                   return_sequences=False)

def connect_encoder():
    # Start the neural network with its input-layer.
    net = encoder_input

    # Connect the embedding-layer.
    net = encoder_embedding(net)
​
    # Connect all the GRU-layers.
    net = encoder_gru1(net)
    net = encoder_gru2(net)
    net = encoder_gru3(net)
​
    # This is the output of the encoder.
    encoder_output = net

    return encoder_output

encoder_output = connect_encoder()

decoder_initial_state = Input(shape=(state_size,),
                              name='decoder_initial_state')

decoder_input = Input(shape=(None, ), name='decoder_input')

decoder_embedding = Embedding(input_dim=num_words,
                              output_dim=embedding_size,
                              name='decoder_embedding')

decoder_gru1 = GRU(state_size, name='decoder_gru1',
                   return_sequences=True)
decoder_gru2 = GRU(state_size, name='decoder_gru2',
                   return_sequences=True)
decoder_gru3 = GRU(state_size, name='decoder_gru3',
                   return_sequences=True)

decoder_dense = Dense(num_words,
                      activation='linear',
                      name='decoder_output')

def connect_decoder(initial_state):
    # Start the decoder-network with its input-layer.
    net = decoder_input
​
    # Connect the embedding-layer.
    net = decoder_embedding(net)

    # Connect all the GRU-layers.
    net = decoder_gru1(net, initial_state=initial_state)
    net = decoder_gru2(net, initial_state=initial_state)
    net = decoder_gru3(net, initial_state=initial_state)
​
    # Connect the final dense layer that converts to
    # one-hot encoded arrays.
    decoder_output = decoder_dense(net)

    return decoder_output

decoder_output = connect_decoder(initial_state=encoder_output)

model_train = Model(inputs=[encoder_input, decoder_input],
                    outputs=[decoder_output])

model_encoder = Model(inputs=[encoder_input],
                      outputs=[encoder_output])

decoder_output = connect_decoder(initial_state=decoder_initial_state)

model_decoder = Model(inputs=[decoder_input, decoder_initial_state],
                      outputs=[decoder_output])

# model_train.compile(optimizer=optimizer,
#                     loss='sparse_categorical_crossentropy')

def sparse_cross_entropy(y_true, y_pred):
    """
    Calculate the cross-entropy loss between y_true and y_pred.

    y_true is a 2-rank tensor with the desired output.
    The shape is [batch_size, sequence_length] and it
    contains sequences of integer-tokens.
​
    y_pred is the decoder's output which is a 3-rank tensor
    with shape [batch_size, sequence_length, num_words]
    so that for each sequence in the batch there is a one-hot
    encoded array of length num_words.
    """
​
    # Calculate the loss. This outputs a
    # 2-rank tensor of shape [batch_size, sequence_length]
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                          logits=y_pred)
​
    # Keras may reduce this across the first axis (the batch)
    # but the semantics are unclear, so to be sure we use
    # the loss across the entire 2-rank tensor, we reduce it
    # to a single scalar with the mean function.
    loss_mean = tf.reduce_mean(loss)
​
    return loss_mean

optimizer = RMSprop(lr=1e-3)

decoder_target = tf.placeholder(dtype='int32', shape=(None, None))

model_train.compile(optimizer=optimizer,
                    loss=sparse_cross_entropy,
                    target_tensors=[decoder_target])

path_checkpoint = '21_checkpoint.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)

callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=3, verbose=1)

callback_tensorboard = TensorBoard(log_dir='./21_logs/',
                                   histogram_freq=0,
                                   write_graph=False)

callbacks = [callback_early_stopping,
             callback_checkpoint,
             callback_tensorboard]

try:
    model_train.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)

x_data = \
{
    'encoder_input': encoder_input_data,
    'decoder_input': decoder_input_data
}

y_data = \
{
    'decoder_output': decoder_output_data
}

validation_split = 10000 / len(encoder_input_data)
validation_split

model_train.fit(x=x_data,
                y=y_data,
                batch_size=640,
                epochs=10,
                validation_split=validation_split,
                callbacks=callbacks)

def translate(input_text, true_output_text=None):
    """Translate a single text-string."""
​
    # Convert the input-text to integer-tokens.
    # Note the sequence of tokens has to be reversed.
    # Padding is probably not necessary.
    input_tokens = tokenizer_src.text_to_tokens(text=input_text,
                                                reverse=True,
                                                padding=True)

    # Get the output of the encoder's GRU which will be
    # used as the initial state in the decoder's GRU.
    # This could also have been the encoder's final state
    # but that is really only necessary if the encoder
    # and decoder use the LSTM instead of GRU because
    # the LSTM has two internal states.
    initial_state = model_encoder.predict(input_tokens)
​
    # Max number of tokens / words in the output sequence.
    max_tokens = tokenizer_dest.max_tokens
​
    # Pre-allocate the 2-dim array used as input to the decoder.
    # This holds just a single sequence of integer-tokens,
    # but the decoder-model expects a batch of sequences.
    shape = (1, max_tokens)
    decoder_input_data = np.zeros(shape=shape, dtype=np.int)
​
    # The first input-token is the special start-token for 'ssss '.
    token_int = token_start
​
    # Initialize an empty output-text.
    output_text = ''
​
    # Initialize the number of tokens we have processed.
    count_tokens = 0
​
    # While we haven't sampled the special end-token for ' eeee'
    # and we haven't processed the max number of tokens.
    while token_int != token_end and count_tokens < max_tokens:
        # Update the input-sequence to the decoder
        # with the last token that was sampled.
        # In the first iteration this will set the
        # first element to the start-token.
        decoder_input_data[0, count_tokens] = token_int
​
        # Wrap the input-data in a dict for clarity and safety,
        # so we are sure we input the data in the right order.
        x_data = \
        {
            'decoder_initial_state': initial_state,
            'decoder_input': decoder_input_data
        }
​
        # Note that we input the entire sequence of tokens
        # to the decoder. This wastes a lot of computation
        # because we are only interested in the last input
        # and output. We could modify the code to return
        # the GRU-states when calling predict() and then
        # feeding these GRU-states as well the next time
        # we call predict(), but it would make the code
        # much more complicated.
​
        # Input this data to the decoder and get the predicted output.
        decoder_output = model_decoder.predict(x_data)
​
        # Get the last predicted token as a one-hot encoded array.
        token_onehot = decoder_output[0, count_tokens, :]

        # Convert to an integer-token.
        token_int = np.argmax(token_onehot)
​
        # Lookup the word corresponding to this integer-token.
        sampled_word = tokenizer_dest.token_to_word(token_int)
​
        # Append the word to the output-text.
        output_text += " " + sampled_word
​
        # Increment the token-counter.
        count_tokens += 1
​
    # Sequence of tokens output by the decoder.
    output_tokens = decoder_input_data[0]

    # Print the input-text.
    print("Input text:")
    print(input_text)
    print()
​
    # Print the translated output-text.
    print("Translated text:")
    print(output_text)
    print()
​
    # Optionally print the true translated text.
    if true_output_text is not None:
        print("True output text:")
        print(true_output_text)
        print()

idx = 3
translate(input_text=data_src[idx],
          true_output_text=data_dest[idx])

idx = 4
translate(input_text=data_src[idx],
          true_output_text=data_dest[idx])

idx = 3
translate(input_text=data_src[idx] + data_src[idx+1],
          true_output_text=data_dest[idx] + data_dest[idx+1])

idx = 3
translate(input_text=data_src[idx+1] + data_src[idx],
          true_output_text=data_dest[idx+1] + data_dest[idx])

translate(input_text="der var engang et land der hed Danmark",
          true_output_text='Once there was a country named Denmark')

translate(input_text="Idag kan man læse i avisen at Danmark er blevet fornuftigt",
          true_output_text="Today you can read in the newspaper that Denmark has become sensible.")

translate(input_text="Hvem spæner ud af en butik og tygger de stærkeste bolcher?",
          true_output_text="Who runs out of a shop and chews the strongest bon-bons?")
