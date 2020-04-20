import argparse
from collections import Counter
import os
import numpy as np
import tensorflow as tf

def get_args():
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--pe_target', type=int, default=230, help='Target Positional Encoding')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/train_bt_st_5_upsample_redo__acc_', help='load checkpoint path')
    parser.add_argument('--npz_path', type=str, default='./checkpoints/train_bt_st_5_upsample_redo__acc_/data_and_vocab_bt_st_upsample_.npz', help='load checkpoint path')
    parser.add_argument('--experiment', type=str, default="experiments", help='experiment label')
    parser.add_argument('--data_path', type=str, default="/home/guest159/project2/data", help='source data path')
    parser.add_argument('--bt', action="store_true", help='source data path')
    parser.add_argument('--st', action="store_true", help='source data path')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of model')
    parser.add_argument('--epochs', type=int, default=2, help='batch size of model')
    parser.add_argument('--num_layers', type=int, default=2, help='num of of model')
    parser.add_argument('--d_model', type=int, default=1024, help='batch size of model')
    parser.add_argument('--num_heads', type=int, default=8, help='batch size of model')
    parser.add_argument('--dff', type=int, default=1024, help='batch size of model')
    parser.add_argument('--p_wd_st', type=float, default=0.3, help='batch size of model')
    parser.add_argument('--p_wd_bt', type=float, default=0.1, help='batch size of model')
    parser.add_argument('--dropout_rate', type=float, default=0.4, help='batch size of model')
    parser.add_argument('--start', type=int, default=0, help='batch size of model')
    parser.add_argument('--end', type=int, default=50000, help='batch size of model')
    # parser.add_argument
    args = parser.parse_args()
    return args

def read_file(data_path,path):
    abs_path = os.path.join(data_path,path)
    with open(abs_path,"r") as f:
        data = f.read().strip()
    return data

def read_file_and_process(data_path,path,limit_vocab=10000):
    data = read_file(data_path,path)
    data_new = data.replace("\n", " <eos> ").split()
    vocab = list(set(data_new))
    counter = Counter(data_new)
    counter.update({"<unk>":0}) # artificial count
    counter.update({"<start>":0}) # artificial count
    # most_common = counter.most_common(limit_vocab)
    lang_vocab = list(counter.keys())
    # lang_vocab = [i for i,j in most_common]
    word2id = {}
    id2word = {}
    # start enumerate from 1 so that 0 is reserved for padding seqs 
    # for i, (w,count) in enumerate(most_common, start=1):
    for i, w in enumerate(lang_vocab, start=1):
        word2id[w] = i
        id2word[i] = w
    return data, lang_vocab, word2id, id2word

def write_to_file(all_preds,french_word2id,french_id2word,filename):
    translated_sentences = []
    for k in all_preds:
        for i in k:
            sentence_french = []
            for j in i.numpy()[1:]:
                if j==0 or j==french_word2id["<eos>"]:
                    break
                sentence_french.append(french_id2word[j])
            sentence_french = " ".join(sentence_french)
            translated_sentences.append(sentence_french)
    translated_sentences = "\n".join(translated_sentences)+"\n"
    with open(filename,"a+") as f:
        f.write(translated_sentences)

def transform_data(lang, word2id):
  lines = lang.split("\n")
  
  data = []

  for line in lines:
    line2id = [word2id["<start>"]]
    for word in line.split():
      try:
        line2id.append(word2id[word])
      except:
        line2id.append(word2id["<unk>"])
    line2id.append(word2id["<eos>"])
    data.append(line2id)

  print(len(data))
  return data

class DatasetGenerator_ST(tf.data.Dataset):
  def _generator():
    inp_pad = tf.keras.preprocessing.sequence.pad_sequences(data_english_monolingual, padding='post').shape[1]
    tar_pad = tf.keras.preprocessing.sequence.pad_sequences(data_french_st, padding='post').shape[1]
    indexes = np.arange(len(data_french_st))
    np.random.shuffle(indexes)
    data1 = np.array(data_english_monolingual)[indexes]
    data2 = np.array(data_french_st)[indexes]
    for i in range(len(data2)):
      tar = data2[i]
      tar = np.pad(tar, (0,tar_pad-len(tar)))
      aug = data1[i]
      if np.random.choice(['drop','swap']) == 'drop':
        drop_idxs = np.random.binomial(1,p_wd_st,len(aug))
        drop_idxs = np.where(drop_idxs==1)
        aug = np.delete(aug,drop_idxs)
      else:
        swap_idx = np.random.choice(np.arange(1,len(aug)))
        tmp = aug[swap_idx]
        aug[swap_idx] = aug[swap_idx-1]
        aug[swap_idx-1] = tmp
      aug = np.pad(aug, (0,inp_pad-len(aug)))
      yield aug, tar

  def __new__(cls):
      return tf.data.Dataset.from_generator(
          cls._generator,
          output_types=(tf.dtypes.int32,tf.dtypes.int32),
          output_shapes=(None,None)
      )

class DatasetGenerator_BT(tf.data.Dataset):
  def _generator():
    inp_pad = tf.keras.preprocessing.sequence.pad_sequences(data_english_bt, padding='post').shape[1]
    tar_pad = tf.keras.preprocessing.sequence.pad_sequences(data_french_monolingual, padding='post').shape[1]
    indexes = np.arange(len(data_french_monolingual))
    np.random.shuffle(indexes)
    data1 = np.array(data_english_bt)[indexes]
    data2 = np.array(data_french_monolingual)[indexes]
    for i in range(len(data2)):
      tar = data2[i]
      tar = np.pad(tar, (0,tar_pad-len(tar)))
      aug = data1[i]
      if np.random.choice(['drop','swap']) == 'drop':
        drop_idxs = np.random.binomial(1,p_wd_bt,len(aug))
        drop_idxs = np.where(drop_idxs==1)
        aug = np.delete(aug,drop_idxs)
      else:
        swap_idx = np.random.choice(np.arange(1,len(aug)))
        tmp = aug[swap_idx]
        aug[swap_idx] = aug[swap_idx-1]
        aug[swap_idx-1] = tmp
      aug = np.pad(aug, (0,inp_pad-len(aug)))
      yield aug, tar

  def __new__(cls,data1,data2,p):
      return tf.data.Dataset.from_generator(
          cls._generator,
          output_types=(tf.dtypes.int32,tf.dtypes.int32),
          output_shapes=(None,None)
      )

class DatasetGenerator(tf.data.Dataset):
  def _generator(data_1,data_2,p_wd):
    inp_pad = tf.keras.preprocessing.sequence.pad_sequences(data_1, padding='post').shape[1]
    tar_pad = tf.keras.preprocessing.sequence.pad_sequences(data_2, padding='post').shape[1]
    indexes = np.arange(len(data_2))
    np.random.shuffle(indexes)
    data1 = np.array(data_1)[indexes]
    data2 = np.array(data_2)[indexes]
    for i in range(len(data2)):
      tar = data2[i]
      tar = np.pad(tar, (0,tar_pad-len(tar)))
      aug = data1[i]
      if np.random.choice(['drop','swap']) == 'drop':
        drop_idxs = np.random.binomial(1,p_wd,len(aug))
        drop_idxs = np.where(drop_idxs==1)
        aug = np.delete(aug,drop_idxs)
      else:
        swap_idx = np.random.choice(np.arange(1,len(aug)))
        tmp = aug[swap_idx]
        aug[swap_idx] = aug[swap_idx-1]
        aug[swap_idx-1] = tmp
      aug = np.pad(aug, (0,inp_pad-len(aug)))
      yield aug, tar

  def __new__(cls,data_1,data_2,p_wd):
      return tf.data.Dataset.from_generator(
          lambda: cls._generator(data_1,data_2,p_wd),
          output_types=(tf.dtypes.int32,tf.dtypes.int32),
          output_shapes=(None,None)
      )