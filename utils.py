
import io
import unicodedata
import tensorflow as tf
import re


def create_dataset(path_source, path_destination, num_examples):
  lines_source = io.open(path_source, encoding='UTF-8').read().strip().split('\n')
  lines_destination = io.open(path_destination, encoding='UTF-8').read().strip().split('\n')

  word_pairs = []


  for index, l in enumerate(lines_source[:num_examples]):
      word_source = preprocess_sentence(lines_source[index])
      word_destination = preprocess_sentence(lines_destination[index])
      word_pairs.append((word_source, word_destination))

  return zip(*word_pairs)

def unicode_to_ascii(s):
  return ''.join(c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
  w = unicode_to_ascii(w.lower().strip())

  w = re.sub(r"([?.!,¿])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)

  w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

  w = w.strip()

  w = '<start> ' + w + ' <end>'
  return w


def max_length(tensor):
  return max(len(t) for t in tensor)

def tokenize(lang):
  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='', oov_token='UNK')
  lang_tokenizer.fit_on_texts(lang)

  tensor = lang_tokenizer.texts_to_sequences(lang)

  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')

  return tensor, lang_tokenizer

def load_dataset(path_source, path_destination, num_examples=None):
  inp_lang, targ_lang = create_dataset(path_source, path_destination, num_examples)

  input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
  target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

  return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer



def load_dataset_2_files(path_source, path_destination, num_examples=None, ratio=1):
  inp_lang_0, targ_lang_1 = create_dataset(path_source[0], path_destination[0], num_examples)
  inp_lang_1, targ_lang_0 = create_dataset(path_source[1], path_destination[1], num_examples)

  inp_lang = []
  targ_lang = []

  for index in range(num_examples):
    inp_lang.append(inp_lang_0[index])
    targ_lang.append(targ_lang_0[index])

    if index % ratio == 0:
       inp_lang.append(inp_lang_1[int(index/ratio)])
       targ_lang.append(targ_lang_1[int(index/ratio)])


  input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
  target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

  return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


def convert(lang, tensor):
  for t in tensor:
    if t!=0:
      print ("%d ----> %s" % (t, lang.index_word[t]))
