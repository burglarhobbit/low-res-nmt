def generate_predictions_bs(inp_sentences, n_sentences):

  # pe_target set to 50 because OOM problem. 
  # It can be removed.

  pe_target = 50

  # This part were in the original code: generate_predictions
  # Should be tested in both conditions

  if len(inp_sentences.get_shape())==1:
    encoder_input = tf.expand_dims(inp_sentences, 0)
    decoder_input = [french_word2id["<start>"]]

  else:
    encoder_input = inp_sentences
    decoder_input = [french_word2id["<start>"]]*inp_sentences.get_shape()[0]

  # This is the batch for each generation

  batch_gener = len(decoder_input)

  # n_sentence is the number of different sentences that we would like to generate

  for s in range(n_sentences):

  # This part were in the original code: generate_predictions
  # For every sentence the output is initiate here
  # Should be tested in both conditions

    if len(inp_sentences.get_shape())==1:
      output = tf.expand_dims(decoder_input, 0) 
    else:
      output = tf.expand_dims(decoder_input, -1)
    
  # For every sentence the output_score is initiate here with the same dimension of output
  # In output_score is storage the score of each word of pe_target

    output_score = tf.zeros(shape=tf.shape(output).numpy(), dtype=tf.dtypes.float32, name=None)

    for i in range(pe_target):

  # This part were in the original code: generate_predictions

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)
      
        predictions, attention_weights = transformer(encoder_input, 
                                                    output,
                                                    False,
                                                    enc_padding_mask,
                                                    combined_mask,
                                                    dec_padding_mask)

  # 1) sort - to obtain the score in descending order
  # 2) argsort - to obtain the position where score in descending order

        scores_sorted = tf.sort(predictions,axis=-1,direction='DESCENDING',name=None)
        preditions_sorted = tf.argsort(predictions,axis=-1,direction='DESCENDING',stable=False,name=None)

  # This is the condition where we are going to take the second most likely word
  # When s == 0, all the words are most likely. The result is the same as generate_predictions
  # When s != 0, it's going to change the first word for s == 1, the second word for s == 2,  and so on.

        if i == (s - 1) and s != 0:
            scores = tf.expand_dims(scores_sorted[:, -1, 1], 1)
            predicted_id = tf.expand_dims(preditions_sorted[:, -1, 1], 1)
        else:
            scores = tf.expand_dims(scores_sorted[:, -1, 0], 1)
            predicted_id = tf.expand_dims(preditions_sorted[:, -1, 0], 1)

        output = tf.concat([output, predicted_id], axis=-1)
        output_score = tf.concat([output_score, scores], axis=-1)

  # https://stackoverflow.com/questions/44657388/how-to-replace-certain-values-in-tensorflow-tensor-with-the-values-of-the-other
  # This part is going to calculate the log of each score when it's positive, otherwise it gives 0
  # The output_ave_score is the average score for each sentence

    condition = tf.greater(output_score, 0)
    case_true = tf.math.log(output_score)
    case_false = 0
    a_m = tf.where(condition, case_true, case_false)
    output_ave_score = tf.math.divide(tf.reduce_sum(a_m, axis=1), tf.math.count_nonzero(a_m, dtype=float, axis=1))

  # The tensors with _beam_search has all the sentences (s)

    if s == 0:
        output_beam_search = output
        output_score_beam_search = output_ave_score
    else:
        output_beam_search = tf.concat([output_beam_search, output], axis=-1)
        output_score_beam_search = tf.concat([output_score_beam_search, output_ave_score], axis=-1)

  # This is a reshape:
  # Output_beam_search - (batch, number of sentences, pe_target+1)
  # Output_score_beam_search - (batch, number of sentences) - 1 score per sentence

  output_beam_search = tf.reshape(output_beam_search, [batch_gener,  n_sentences, pe_target + 1])
  output_score_beam_search = tf.reshape(output_score_beam_search, [batch_gener, n_sentences])

  # This part obtains the indices of the the chosen sentence

  indices = tf.expand_dims(tf.argmax(output_score_beam_search, axis=-1), axis=-1)
  # print(indices)

  # it returns the the sentence with the chosen index

  return tf.gather_nd(output_beam_search, indices, batch_dims=1)