from pathlib import Path
import os
import sys
from collections import Counter
import numpy as np
import time
from tqdm import tqdm
from utils import *

import tensorflow as tf
from transformer import Transformer, CustomSchedule, create_masks

np.random.seed(8080)
tf.random.set_seed(8080)

# signature for tf train and val function, needed for using the function with variable batch sizes without creating a different node in the tf graph each time
train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
]


class TrainManager:
    """
    TrainManager handle for efficiently training the transformer model by
    holding instances of different variables.
    """

    def __init__(self, transformer, loss_object, train_loss, train_accuracy, val_loss, val_accuracy, optimizer):
        """
            Args:
                transformer: A transformer.Transformer class object
                loss_object: A tf.keras.losses object to compute loss
                train_loss: A tf.keras.metrics object to store mean loss of train set
                train_accuracy: A tf.keras.metrics.Mean object to store mean accuracy of train set
                val_loss: A tf.keras.metrics.Mean object to store mean loss of validation set
                val_accuracy: A tf.keras.metrics.Mean object to store mean accuracy of validation set
                optimizer: A tf.keras.optimizers.Adam object to optimizer gradients over the loss
            Returns:
                None
            """
        self.transformer = transformer
        self.loss_object = loss_object
        self.val_loss = val_loss
        self.val_accuracy = val_accuracy
        self.train_loss = train_loss
        self.train_accuracy = train_accuracy
        self.optimizer = optimizer

    def loss_function(self, real, pred):
        """Calculates the masked crossentropy loss

        Args:
          real (tf.Tensor, dtype int32): the actual token ids
          pred (tf.Tensor, dtype float32): predictions from the transformer decoder

        Returns:
          float: loss value
        """
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    @tf.function(input_signature=train_step_signature)
    def train_step(self, inp, tar):
        """A single gradient update over batch (uses teacher forcing)

        Args:
          inp (tf.Tensor, dtype int32): the source language token ids
          tar (tf.Tensor, dtype int32): the target language token ids

        Returns: None

        """
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = self.transformer(inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
            loss = self.loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, self.transformer.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(tar_real, predictions)

    @tf.function(input_signature=train_step_signature)
    def val_step(self, inp, tar):
        """validation loss and accuracy calculation over batch (uses teacher forcing)

        Args:
          inp (tf.Tensor, dtype int32): the source language token ids
          tar (tf.Tensor, dtype int32): the target language token ids

        Returns: None

        """
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        predictions, _ = self.transformer(inp, tar_inp,
                                          False,
                                          enc_padding_mask,
                                          combined_mask,
                                          dec_padding_mask)
        loss = self.loss_function(tar_real, predictions)

        self.val_loss(loss)
        self.val_accuracy(tar_real, predictions)

    def generate_predictions(self, inp_sentences, french_word2id, pe_target):
        """Generate target language translations (autoregressive)

        Args:
          inp_sentences (tf.Tensor, dtype int32): the source language token ids
          french_word2id (dict): target language vocabulary (word to id mapping) dictionary
          pe_target (int): sequence length to generate

        Returns:
          output (tf.Tensor, dtype int32): predicted target language token ids
          attention_weights (tf.Tensor, dtype float32): decoder attention weights
        """
        if len(inp_sentences.get_shape()) == 1:
            encoder_input = tf.expand_dims(inp_sentences, 0)
            decoder_input = [french_word2id["<start>"]]
            output = tf.expand_dims(decoder_input, 0)

        else:
            encoder_input = inp_sentences
            decoder_input = [french_word2id["<start>"]] * inp_sentences.get_shape()[0]
            output = tf.expand_dims(decoder_input, -1)

        for i in range(pe_target):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                encoder_input, output)

            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = self.transformer(encoder_input, output, False,
                                                              enc_padding_mask, combined_mask, dec_padding_mask)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # # return the result if all the seqs has the end token
            if tf.reduce_sum(
                tf.cast((
                    tf.reduce_sum(
                        tf.cast(
                            output == french_word2id["<eos>"], tf.float32
                        ), axis=1
                    ) > 0
                ), tf.float32)
            ) == inp_sentences.get_shape()[0]:
                return output, attention_weights

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)
        return output, attention_weights

    def generate_and_eval(self, tensor_val, val_lang2, word2id, id2word, pe_target):
        """Generate translation predictions and evaluate BLEU score

        Args:
          tensor_val (tf.data.Dataset): validation dataset
          val_lang2 (string): file containing target language text
          word2id (dict): target language dictionary (word to id mapping)
          id2word (dict): target language dictionary (id to word mapping)
          pe_target (int): sequence length to generate

        Returns: None

        """
        from evaluator import compute_bleu
        all_preds = []
        out_file = "predictions.txt"
        for (batch_i, (inp, tar)) in tqdm(enumerate(tensor_val)):
            all_preds = []
            preds, attention = self.generate_predictions(inp, word2id, pe_target)
            all_preds.append(preds)
            write_to_file(all_preds, word2id, id2word, out_file)
        compute_bleu(out_file, val_lang2, False)


def main(args):
    """
    Main function to perform the training task

    Args:
      args: An argparse object containing processed arguments
    Returns:
      None
    """
    data_path = args.data_path

    print(data_path)
    print(os.listdir(data_path))
    print("Tensorflow version " + tf.__version__)

    train_lang1 = "train/split_train.lang1"
    train_lang2 = "train/split_train.lang2"
    val_lang1 = "val/split_val.lang1"
    val_lang2 = "val/split_val.lang2"

    # load data and create vocab
    english, english_vocab, english_word2id, english_id2word = read_file_and_process(data_path, train_lang1, limit_vocab=9000)
    french, french_vocab, french_word2id, french_id2word = read_file_and_process(data_path, train_lang2, limit_vocab=10000)
    english_val = read_file(data_path, val_lang1)
    french_val = read_file(data_path, val_lang2)

    # read monolingual data and synthetic bitext
    french_st = read_file(data_path, "predictions/predictions_english_st_regex.txt")
    print(len(french_st.split("\n")), french_st[:200])

    english_monolingual = read_file(data_path, "unaligned_tokenized_rempunc.en").lower()
    print(len(english_monolingual.split("\n")), english_monolingual[:200])

    english_bt = read_file(data_path, "predictions/predictions_french_bt_regex.txt")
    print(len(english_bt.split("\n")), english_bt[:200])

    french_monolingual = read_file(data_path, "unaligned_tokenized.fr")
    print(len(french_monolingual.split("\n")), french_monolingual[:200])

    print(len(english_word2id), len(english_id2word), len(french_word2id), len(french_id2word))

    def transform_data_2(english, french):
        data_1 = transform_data(english, english_word2id)
        data_2 = transform_data(french, french_word2id)
        return data_1, data_2

    # transform all data to vocab token ids
    data_english, data_french = transform_data_2(english, french)
    data_english_val, data_french_val = transform_data_2(english_val, french_val)

    # align latge corpus monolingual data with the corresponding synthetic bitext
    monolingual_start_index = args.start
    data_english_monolingual, data_french_st = transform_data_2(english_monolingual, french_st)
    data_english_monolingual = data_english_monolingual[monolingual_start_index:len(data_french_st) + monolingual_start_index]

    data_english_bt, data_french_monolingual = transform_data_2(english_bt, french_monolingual)
    data_french_monolingual = data_french_monolingual[monolingual_start_index:len(data_english_bt) + monolingual_start_index]

    print(english_id2word[54], len(data_english), len(data_french), len(data_english_val), len(data_french_val), len(data_english_monolingual), len(data_french_st), len(data_english_bt), len(data_french_monolingual))

    np.savez("data_and_vocab_%s.npz" % args.experiment, data_english=data_english, data_french=data_french, data_english_val=data_english_val, data_french_val=data_french_val,
             data_english_monolingual=data_english_monolingual, data_french_st=data_french_st,
             data_english_bt=data_english_bt, data_french_monolingual=data_french_monolingual,
             english_word2id=english_word2id, english_id2word=english_id2word, french_word2id=french_word2id, french_id2word=french_id2word)

    BUFFER_SIZE = len(data_english)
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    print("No. of batches: ", np.ceil(len(data_english_monolingual) / BATCH_SIZE))
    print("No. of batches: ", np.ceil(len(data_english) / BATCH_SIZE))
    repeat_factor = len(data_english_monolingual) // len(data_english) + 1  # upsampling rate
    print(repeat_factor)

    # transformer hyperparams
    num_layers = args.num_layers
    d_model = args.d_model
    dff = args.dff
    num_heads = args.num_heads
    input_vocab_size = len(english_vocab) + 1
    target_vocab_size = len(french_vocab) + 1
    dropout_rate = args.dropout_rate
    p_wd_st = args.p_wd_st
    p_wd_bt = args.p_wd_bt
    pe_input = max(max([len(i) for i in data_english]), max([len(i) for i in data_english_val]), max([len(i) for i in data_english_monolingual]), max([len(i) for i in data_english_bt]))
    pe_target = max(max([len(i) for i in data_french]), max([len(i) for i in data_french_val]), max([len(i) for i in data_french_st]), max([len(i) for i in data_french_monolingual]))

    # pe_input = 200
    # pe_target = 230
    print(pe_input, pe_target)

    tensor_train = tf.data.Dataset.from_tensor_slices((
        tf.keras.preprocessing.sequence.pad_sequences(data_english, padding='post'),
        tf.keras.preprocessing.sequence.pad_sequences(data_french, padding='post')
    )).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=False)
    tensor_val = tf.data.Dataset.from_tensor_slices((
        tf.keras.preprocessing.sequence.pad_sequences(data_english_val, padding='post'),
        tf.keras.preprocessing.sequence.pad_sequences(data_french_val, padding='post')
    )).batch(BATCH_SIZE, drop_remainder=False)

    tensor_st = DatasetGenerator(english_monolingual, french_st, p_wd_st).batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.experimental.AUTOTUNE)
    tensor_bt = DatasetGenerator(english_bt, french_monolingual, p_wd_bt).batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.experimental.AUTOTUNE)

    transformer = Transformer(
        num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
        input_vocab_size=input_vocab_size, target_vocab_size=target_vocab_size,
        pe_input=pe_input, pe_target=pe_target, rate=dropout_rate)

    # sample input output for transformer for verifying model summary and shapes
    temp_input = tf.random.uniform((BATCH_SIZE, pe_input), dtype=tf.int64, minval=0, maxval=200)
    temp_target = tf.random.uniform((BATCH_SIZE, pe_target), dtype=tf.int64, minval=0, maxval=200)

    fn_out, _ = transformer(temp_input, temp_target, training=False,
                            enc_padding_mask=None,
                            look_ahead_mask=None,
                            dec_padding_mask=None)

    print(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)

    transformer.summary()

    learning_rate = CustomSchedule(d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    train_loss = tf.keras.metrics.Mean(name='loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')

    val_loss = tf.keras.metrics.Mean(name='loss')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='val_accuracy')

    train_manager = TrainManager(transformer, loss_object, train_loss, train_accuracy, val_loss, val_accuracy, optimizer)
    experiment_number = args.experiment

    # for saving models with best validation loss and validation accuracy respectively
    checkpoint_path = data_path + "/checkpoints/train" + experiment_number
    checkpoint_path_acc = data_path + "/checkpoints/train" + experiment_number + "_acc_"

    ckpt = tf.train.Checkpoint(transformer=train_manager.transformer, optimizer=train_manager.optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)
    ckpt_manager_acc = tf.train.CheckpointManager(ckpt, checkpoint_path_acc, max_to_keep=1)

    # for tensorboard
    writer_train = tf.summary.create_file_writer("log_dir/" + experiment_number + "_train")
    writer_val = tf.summary.create_file_writer("log_dir/" + experiment_number + "_val")

    # Start training
    best_val_loss = np.inf
    best_val_acc = 0

    for epoch in range(EPOCHS):
        start = time.time()

        train_manager.train_loss.reset_states()
        train_manager.train_accuracy.reset_states()

        if args.st:
            print("training ST data")
            tensor_st = DatasetGenerator(data_english_monolingual, data_french_st, p_wd_st).batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.experimental.AUTOTUNE)

            for (batch, (inp, tar)) in tqdm(enumerate(tensor_st), total=len(data_english_monolingual) // BATCH_SIZE + 1):
                train_manager.train_step(inp, tar)
                if batch % 100 == 0:
                    print('Epoch {} Batch {} Training Loss {:.4f} Accuracy {:.4f}'.format(
                        epoch + 1, batch, train_manager.train_loss.result(), train_manager.train_accuracy.result()))
        if args.bt:
            print("training BT data")
            tensor_bt = DatasetGenerator(data_english_bt, data_french_monolingual, p_wd_bt).batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.experimental.AUTOTUNE)

            for (batch, (inp, tar)) in tqdm(enumerate(tensor_bt), total=len(data_english_bt) // BATCH_SIZE + 1):
                train_manager.train_step(inp, tar)
                if batch % 100 == 0:
                    print('Epoch {} Batch {} Training Loss {:.4f} Accuracy {:.4f}'.format(
                        epoch + 1, batch, train_manager.train_loss.result(), train_manager.train_accuracy.result()))

        for iteration_i in range(repeat_factor):

            train_manager.train_loss.reset_states()
            train_manager.train_accuracy.reset_states()

            train_manager.val_loss.reset_states()
            train_manager.val_accuracy.reset_states()

            print("training Parallel data")

            for (batch, (inp, tar)) in tqdm(enumerate(tensor_train)):
                train_step(inp, tar)
                if batch % 50 == 0:
                    print('Epoch {} iteration_i {} Batch {} Training Loss {:.4f} Accuracy {:.4f}'.format(
                        epoch + 1, iteration_i, batch, train_manager.train_loss.result(), train_manager.train_accuracy.result()))

            print('Epoch {} iteration_i {} Training Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                                                        iteration_i,
                                                                                        train_manager.train_loss.result(),
                                                                                        train_manager.train_accuracy.result()))
            print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

            with writer_train.as_default():
                tf.summary.scalar('train_loss', train_manager.train_loss.result(), step=epoch)

            print("validating")
            for (batch, (inp, tar)) in enumerate(tensor_val):
                train_manager.val_step(inp, tar)

            print('Epoch {} iteration_i {} Validation Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                                                          iteration_i,
                                                                                          train_manager.val_loss.result(),
                                                                                          train_manager.val_accuracy.result()))
            if best_val_loss > train_manager.val_loss.result():
                best_val_loss = train_manager.val_loss.result()
                ckpt_save_path = ckpt_manager.save()
                print('Saving checkpoint for epoch {} iteration_i {} at {}'.format(epoch + 1,
                                                                                   iteration_i,
                                                                                   ckpt_save_path))
            if best_val_acc < train_manager.val_accuracy.result():
                best_val_acc = train_manager.val_accuracy.result()
                ckpt_save_path = ckpt_manager_acc.save()
                print('Saving checkpoint for epoch {} iteration_i {} at {}'.format(epoch + 1,
                                                                                   iteration_i,
                                                                                   ckpt_save_path))

            with writer_val.as_default():
                tf.summary.scalar('val_loss', train_manager.val_loss.result(), step=epoch)

    """Evaluate best model"""
    # load model
    print(ckpt_manager_acc.checkpoints)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager_acc.latest_checkpoint:
        ckpt.restore(ckpt_manager_acc.latest_checkpoint)
        print('Latest checkpoint restored!!')

    train_manager.val_loss.reset_states()
    train_manager.val_accuracy.reset_states()

    for (batch, (inp, tar)) in tqdm(enumerate(tensor_val)):
        train_manager.val_step(inp, tar)

    print('Validation Loss {:.4f} Accuracy {:.4f}'.format(train_manager.val_loss.result(), train_manager.val_accuracy.result()))
    train_manager.generate_and_eval(tensor_val, os.path.join(data_path, val_lang2), french_word2id, french_id2word, pe_target)


if __name__ == "__main__":
    args = get_args()
    data_path = args.data_path
    main(args)
