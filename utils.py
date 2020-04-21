import argparse
from collections import Counter
import os
import numpy as np
import tensorflow as tf


def get_args():
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--pe_target', type=int, default=230, help='Target Positional Encoding')
    parser.add_argument('--checkpoint_path', type=str, default='../model/checkpoints/train_bt_st_iter2_upsample_1__acc_', help='load checkpoint path')
    parser.add_argument('--npz_path', type=str, default='../model/data_and_vocab_bt_st_upsample_best.npz', help='npz file path')
    parser.add_argument('--experiment', type=str, default="bt_st_iter2_upsample_1", help='experiment label during training')
    parser.add_argument('--data_path', type=str, default="/home/guest159/project2/data", help='source data path')
    parser.add_argument('--train_lang1', type=str, default="train/split_train.lang1", help='source data path')
    parser.add_argument('--train_lang2', type=str, default="train/split_train.lang2", help='source data path')
    parser.add_argument('--val_lang1', type=str, default="val/split_val.lang1", help='source data path')
    parser.add_argument('--val_lang2', type=str, default="val/split_val.lang2", help='source data path')
    parser.add_argument('--bt', action="store_true", help='use back translation data')
    parser.add_argument('--st', action="store_true", help='use self training data')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of model')
    parser.add_argument('--epochs', type=int, default=2, help='num epochs to train on')
    parser.add_argument('--num_layers', type=int, default=2, help='num layers of transformer model')
    parser.add_argument('--d_model', type=int, default=1024, help='hidden dimension of embedding and attention output')
    parser.add_argument('--num_heads', type=int, default=8, help='num heads of model')
    parser.add_argument('--dff', type=int, default=1024, help='dimension of feed forward layer')
    parser.add_argument('--p_wd_st', type=float, default=0.3, help='probability of word drop on self training')
    parser.add_argument('--p_wd_bt', type=float, default=0.1, help='probability of word drop on back translation')
    parser.add_argument('--dropout_rate', type=float, default=0.4, help='dropout rate')
    parser.add_argument('--start', type=int, default=0, help='start index in input data lines')
    parser.add_argument('--end', type=int, default=50000, help='end index in input data lines')
    # parser.add_argument
    args = parser.parse_args()
    return args


def read_file(data_path, path):
    """
    Reads the content of a file and returns it's text

    Args:
      data_path (str): Directory inside which file exists
      path (str): Relative path of the file w.r.t data_path

    Returns:
      data (str): contents of the file: 'path'
    """
    abs_path = os.path.join(data_path, path)
    with open(abs_path, "r") as f:
        data = f.read().strip()
    return data


def read_file_and_process(data_path, path):
    """
    Reads the content of a file and generates two-way vocabulary
    Args:
      data_path (str): Directory inside which file exists
      path (str): Relative path of the file w.r.t data_path

    Returns:
      data (str): contents of the file: 'path'
      lang_vocab (list): A list of unique vocabulary
      word2id (dict): A dictionary mapping of words to its numerical ids
      id2word (dict): A reverse-dictionary mapping of numerical ids to its respective words
    """
    data = read_file(data_path, path)
    data_new = data.replace("\n", " <eos> ").split()
    vocab = list(set(data_new))
    counter = Counter(data_new)
    counter.update({"<unk>": 0})
    counter.update({"<start>": 0})

    lang_vocab = list(counter.keys())

    word2id = {}
    id2word = {}

    # start enumerate from 1 so that 0 is reserved for padding seqs
    for i, w in enumerate(lang_vocab, start=1):
        word2id[w] = i
        id2word[i] = w
    return data, lang_vocab, word2id, id2word


def write_to_file(all_preds, word2id, id2word, filename):
    """
    Decodes predictions and saves the generated text into a file.
    Args:
      all_preds (list): A lists of lists which contains ids of predicted text in tf.data format
      word2id (dict): A dictionary mapping of words to its numerical ids
      id2word (dict): A reverse-dictionary mapping of numerical ids to its respective words
      filename (str): Filename to save the generated textual predictions in.
    Returns:
      None
    """
    translated_sentences = []
    for k in all_preds:
        for i in k:
            sentence = []
            for j in i.numpy()[1:]:
                if j == 0 or j == word2id["<eos>"]:
                    break
                sentence.append(id2word[j])
            sentence = " ".join(sentence)
            translated_sentences.append(sentence)
    translated_sentences = "\n".join(translated_sentences) + "\n"
    with open(filename, "a+") as f:
        f.write(translated_sentences)


def transform_data(lang, word2id, amount_data_start=None, amount_data_end=None):
    """
    Transforms the textual tokenized data into its respective numerical ids given a word2id mapping
    Args:
      lang (str): full raw tokenized text to be transformed
      word2id (dict): A dictionary mapping of words to its numerical ids
      amount_data_start (int, Optional): starting line index of the data
      amount_data_end (int, Optional): ending line index of the data
    Returns:
      data (list): List of lists contained ids of the tokenized textual data
    """

    lines = lang.split("\n")

    if amount_data_start is not None and amount_data_end is not None:
        lines = lines[amount_data_start:amount_data_end]
    data = []

    for line in lines:
        line2id = [word2id["<start>"]]
        for word in line.split():
            try:
                line2id.append(word2id[word])
            except BaseException:
                line2id.append(word2id["<unk>"])
        line2id.append(word2id["<eos>"])
        data.append(line2id)

    print(len(data))
    return data


class DatasetGenerator(tf.data.Dataset):
    """
    Create a tf.data.Dataset with data augmentation: noise in input sequences
    """
    def _generator(data_1, data_2, p_wd):
        """
        Generator with noising in inputs

        Args:
          data_1 (list): list of lists of source language sequences
          data_2 (list): list of lists of target language sequences
          p_wd (float): probability of word drop

        Returns:
          aug (tf.Tensor): input for transformer
          tar (tf.Tensor): real target for transformer
        """
        inp_pad = tf.keras.preprocessing.sequence.pad_sequences(data_1, padding='post').shape[1]
        tar_pad = tf.keras.preprocessing.sequence.pad_sequences(data_2, padding='post').shape[1]
        indexes = np.arange(len(data_2))
        np.random.shuffle(indexes)
        data1 = np.array(data_1)[indexes]
        data2 = np.array(data_2)[indexes]
        for i in range(len(data2)):
            tar = data2[i]
            tar = np.pad(tar, (0, tar_pad - len(tar)))
            aug = data1[i]
            if np.random.choice(['drop', 'swap']) == 'drop':
                drop_idxs = np.random.binomial(1, p_wd, len(aug))
                drop_idxs = np.where(drop_idxs == 1)
                aug = np.delete(aug, drop_idxs)
            else:
                swap_idx = np.random.choice(np.arange(1, len(aug)))
                tmp = aug[swap_idx]
                aug[swap_idx] = aug[swap_idx - 1]
                aug[swap_idx - 1] = tmp
            aug = np.pad(aug, (0, inp_pad - len(aug)))
            yield aug, tar

    def __new__(cls, data_1, data_2, p_wd):
        return tf.data.Dataset.from_generator(
            lambda: cls._generator(data_1, data_2, p_wd),
            output_types=(tf.dtypes.int32, tf.dtypes.int32),
            output_shapes=(None, None)
        )
