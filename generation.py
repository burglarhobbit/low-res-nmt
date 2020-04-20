##################################################
# DATA GENERATION
##################################################
"""Self-Training Monolingual Data Generation"""
import tensorflow as tf
from transformer import Transformer, CustomSchedule, create_masks
from utils import *
import numpy as np
import os, sys
from tqdm import tqdm


train_step_signature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    ]

class PredictionGenerator:

    def __init__(self,transformer, loss_object, val_loss, val_accuracy):
        self.transformer = transformer
        self.loss_object = loss_object
        self.val_loss = val_loss
        self.val_accuracy = val_accuracy

    def loss_function(self, real, pred):

        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        
        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

    def generate_predictions(self,inp_sentences,french_word2id, pe_target):

        if len(inp_sentences.get_shape())==1:
            encoder_input = tf.expand_dims(inp_sentences, 0)
            decoder_input = [french_word2id["<start>"]]
            output = tf.expand_dims(decoder_input, 0)

        else:
            encoder_input = inp_sentences
            decoder_input = [french_word2id["<start>"]]*inp_sentences.get_shape()[0]
            output = tf.expand_dims(decoder_input, -1)
        
        for i in range(pe_target):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                    encoder_input, output)
        
            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = self.transformer(encoder_input, output, False,
                enc_padding_mask, combined_mask, dec_padding_mask)
        
            # select the last word from the seq_len dimension
            predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

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

        # return tf.squeeze(output, axis=0), attention_weights
        return output, attention_weights

    @tf.function(input_signature=train_step_signature)
    def val_step(self,inp, tar):
        
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
    
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        predictions, _ = self.transformer(inp, tar_inp, False, enc_padding_mask, combined_mask, dec_padding_mask)
        loss = self.loss_function(tar_real, predictions)

        self.val_loss(loss)
        self.val_accuracy(tar_real, predictions)

# similar to train transform but for test and only on one lang with amount_data params
def transform_test_data(lang1, dict_word2id, amount_data_start=None,amount_data_end=None):
    lines = lang1.split("\n")
    if amount_data_start or amount_data_end:
        lines = lines[amount_data_start:amount_data_end]
    data = []

    for line in lines:
        line2id = [dict_word2id["<start>"]]
        for word in line.split():
            try:
                line2id.append(dict_word2id[word])
            except:
                line2id.append(dict_word2id["<unk>"])
        line2id.append(dict_word2id["<eos>"])
        data.append(line2id)
    return data

def main(args):

    npz_path = args.npz_path

    data = np.load(npz_path, allow_pickle=True)
    english_id2word = data["english_id2word"].item()
    english_word2id = data["english_word2id"].item()
    french_word2id = data["french_word2id"].item()
    french_id2word = data["french_id2word"].item()
    data_english_val = data["data_english_val"]
    data_french_val = data["data_french_val"]

    amount_data_start = args.start
    amount_data_end = args.end
    #TODO: 300000-350000
    print(type(amount_data_start))
    print("generating from:%d to %d"%(amount_data_start,amount_data_end))

    english_monolingual = read_file(data_path,"unaligned_tokenized_rempunc.en").lower()
    print(len(english_monolingual.split("\n")), english_monolingual[:200])

    english_monolingual_data = transform_test_data(english_monolingual, english_word2id, amount_data_start, amount_data_end)
    print(len(english_monolingual_data))

    print(max([len(i) for i in english_monolingual_data]))

    pe_input = max([len(i) for i in english_monolingual_data])
    # pe_target = max([len(i) for i in french_monolingual_data])
    pe_target = args.pe_target

    tensor_test = tf.data.Dataset.from_tensor_slices((
            tf.keras.preprocessing.sequence.pad_sequences(english_monolingual_data, padding='post')
    )).batch(args.batch_size, drop_remainder=False)
    tensor_val = tf.data.Dataset.from_tensor_slices((
            tf.keras.preprocessing.sequence.pad_sequences(data_english_val, padding='post'),
            tf.keras.preprocessing.sequence.pad_sequences(data_french_val, padding='post')
    )).batch(args.batch_size, drop_remainder=False)

    input_vocab_size=len(english_id2word) + 1
    target_vocab_size=len(french_id2word) + 1

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    val_loss = tf.keras.metrics.Mean(name='loss')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

    transformer = Transformer(
            num_layers=args.num_layers, d_model=args.d_model, num_heads=args.num_heads, dff=args.dff, 
            input_vocab_size=input_vocab_size, target_vocab_size=target_vocab_size, 
            pe_input=pe_input, pe_target=pe_target, rate=args.dropout_rate)

    prediction_generator = PredictionGenerator(transformer,loss_object,val_loss,val_accuracy)

    learning_rate = CustomSchedule(args.d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    checkpoint_path = args.checkpoint_path
    ckpt = tf.train.Checkpoint(transformer=transformer,
                                                         optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)

    print(ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!!')

    # def get_val_accuracy(tensor_val, val_loss, val_accuracy):

    prediction_generator.val_loss.reset_states()
    prediction_generator.val_accuracy.reset_states()

    for (batch, (inp, tar)) in tqdm(enumerate(tensor_val)):
        prediction_generator.val_step(inp, tar)
        
    print ('Validation Loss {:.4f} Accuracy {:.4f}'.format(val_loss.result(), val_accuracy.result()))

    # get_val_accuracy(tensorflow_val, val_loss, val_accuracy)
    # all_preds = []
    out_file = "predictions_english_monolingual_"+str(amount_data_start)+"_"+str(amount_data_end)+".txt"
    for batch_i, inp in tqdm(enumerate(tensor_test.unbatch().batch(128)),total=len(english_monolingual_data) // 128 + 1):
        all_preds = []
        preds, attention = prediction_generator.generate_predictions(inp, french_word2id, pe_target)
        all_preds.append(preds)
        write_to_file(all_preds, french_word2id, french_id2word, out_file)

if __name__ == "__main__":
    args = get_args()
    data_path = args.data_path
    main(args)