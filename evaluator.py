import argparse
import subprocess
import tempfile


def generate_predictions(input_file_path: str, pred_file_path: str):
    """Generates predictions for the machine translation task (EN->FR).

    You are allowed to modify this function as needed, but one again, you cannot
    modify any other part of this file. We will be importing only this function
    in our final evaluation script. Since you will most definitely need to import
    modules for your code, you must import these inside the function itself.

    Args:
        input_file_path: the file path that contains the input data.
        pred_file_path: the file path where to store the predictions.

    Returns: None

    """

    ##### MODIFY BELOW #####

    import numpy as np
    import tensorflow as tf
    from transformer import Transformer, CustomSchedule, create_masks
    from tqdm import tqdm
    import re

    print("Input file ", input_file_path)

    npz_path = "data_and_vocab_bt_st_upsample_.npz"
    checkpoint_path = "./checkpoints/train_bt_st_5_upsample_redo__acc_"
    pe_target = 230
    d_model = 1024

    # read and transform data to token ids, create tf dataset
    with open(input_file_path, "r") as f:
        english = f.read().strip()

    data = np.load(npz_path, allow_pickle=True)
    english_id2word = data["english_id2word"].item()
    english_word2id = data["english_word2id"].item()
    french_word2id = data["french_word2id"].item()
    french_id2word = data["french_id2word"].item()

    def transform_data(english_lang1):
        english_lines = english_lang1.split("\n")

        data_english = []

        for line in english_lines:
            line2id = [english_word2id["<start>"]]
            for word in line.split():
                try:
                    line2id.append(english_word2id[word])
                except:
                    line2id.append(english_word2id["<unk>"])
            line2id.append(english_word2id["<eos>"])
            data_english.append(line2id)
        print("No. of english sentences: ", len(data_english))
        return data_english

    data_english = transform_data(english)

    tensor_test = tf.data.Dataset.from_tensor_slices((
        tf.keras.preprocessing.sequence.pad_sequences(
            data_english, padding='post')
    )).batch(64, drop_remainder=False)

    # initialize transformer model and ops
    transformer = Transformer(
        num_layers=2, d_model=d_model, num_heads=8, dff=1024,
        input_vocab_size=len(english_id2word)+1, target_vocab_size=len(french_id2word)+1,
        pe_input=200, pe_target=pe_target, rate=0.4
    )

    learning_rate = CustomSchedule(d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)

    # load best model checkpoint
    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(
        ckpt, checkpoint_path, max_to_keep=1)

    print("Loading checkpoint ", ckpt_manager.latest_checkpoint)
    ckpt.restore(ckpt_manager.latest_checkpoint)

    def generate_predictions(inp_sentences):
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
            decoder_input = [french_word2id["<start>"]] * \
                inp_sentences.get_shape()[0]
            output = tf.expand_dims(decoder_input, -1)

        for i in range(pe_target):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                encoder_input, output)

            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = transformer(encoder_input,
                                                         output,
                                                         False,
                                                         enc_padding_mask,
                                                         combined_mask,
                                                         dec_padding_mask)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # # return the result if all the seqs has the end token
            if tf.reduce_sum(tf.cast((tf.reduce_sum(tf.cast(output == french_word2id["<eos>"], tf.float32), axis=1) > 0), tf.float32)) == inp.get_shape()[0]:
                return output, attention_weights

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        # return output, attention_weights
        return output, attention_weights

    # generate and collect predictions from transformer
    all_preds = []
    for batch_i, inp in tqdm(enumerate(tensor_test), total=len(data_english) // 64 + 1):
        preds, attention = generate_predictions(inp)
        all_preds.append(preds)

    translated_sentences = []
    for k in tqdm(all_preds):
        for i in k:
            sentence_french = []
            for j in i.numpy()[1:]:
                if j == 0 or j == french_word2id["<eos>"]:
                    break
                sentence_french.append(french_id2word[j])

            sentence_french = " ".join(sentence_french)

            translated_sentences.append(sentence_french)

    # post-preprocessing
    pattern = re.compile(r'(\b.+\b)\1\b')  # bigram
    out = []
    for line in translated_sentences:
        while pattern.search(line):
            line = pattern.sub(r'\1', line)
        out.append(line)

    translated_sentences = "\n".join(out) + "\n"

    with open(pred_file_path, "w") as f:
        f.write(translated_sentences)

    print("french predictions saved to -->  ", pred_file_path)

    ##### MODIFY ABOVE #####


def compute_bleu(pred_file_path: str, target_file_path: str, print_all_scores: bool):
    """

    Args:
        pred_file_path: the file path that contains the predictions.
        target_file_path: the file path that contains the targets (also called references).
        print_all_scores: if True, will print one score per example.

    Returns: None

    """
    out = subprocess.run(["sacrebleu", "--input", pred_file_path, target_file_path, '--tokenize',
                          'none', '--sentence-level', '--score-only'],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = out.stdout.split('\n')
    if print_all_scores:
        print('\n'.join(lines[:-1]))
    else:
        scores = [float(x) for x in lines[:-1]]
        print('final avg bleu score: {:.2f}'.format(sum(scores) / len(scores)))


def main():
    parser = argparse.ArgumentParser('script for evaluating a model.')
    parser.add_argument('--target-file-path',
                        help='path to target (reference) file', required=True)
    parser.add_argument('--input-file-path',
                        help='path to input file', required=True)
    parser.add_argument('--print-all-scores', help='will print one score per sentence',
                        action='store_true')
    parser.add_argument('--do-not-run-model',
                        help='will use --input-file-path as predictions, instead of running the '
                             'model on it',
                        action='store_true')

    args = parser.parse_args()

    if args.do_not_run_model:
        compute_bleu(args.input_file_path, args.target_file_path,
                     args.print_all_scores)
    else:
        _, pred_file_path = tempfile.mkstemp()
        generate_predictions(args.input_file_path, pred_file_path)
        compute_bleu(pred_file_path, args.target_file_path,
                     args.print_all_scores)


if __name__ == '__main__':
    main()
