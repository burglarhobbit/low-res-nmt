# low-res-nmt

## For evaluation:

please run

`pip install -r requirements.txt`

`python evaluator.py --input-file-path <path-to-test-file> --target-file-path <path-to-target-file>`

## Training our best model

$DATA_PATH=/path/to/data

- `$DATA_PATH` will contain these files
(generated synthetic data from monolingual text; to recreate see constructing the pipelines below):

	- predictions/predictions_english_st_regex.txt 
	- unaligned_tokenized_rempunc.en
	- predictions/predictions_french_bt_regex.txt
	- unaligned_tokenized.fr


- CUDA_VISIBLE_DEVICES="0" python train.py --data_path $DATA_PATH --experiment 1_st --batch_size 64 \
	--num_layer 2 --d_model 1024 --dff 1024 --epochs 3 \
	--p_wd_st 0.3 --p_wd_bt 0.1 --dropout_rate 0.4 --start 200000 \
	--st --bt

### Reconstructing our pipeline (from scratch)

#### Split Data

`$DATA_PATH` should contain these files:
- train.lang1
- train.lang2

- `python split_data.py --data_path $DATA_PATH`

## Generation on Monolingual Data

- `CUDA_VISIBLE_DEVICES="0" python generation.py --checkpoint_path ../model/checkpoints/train_bt_st_iter2_upsample_1__acc_ \
 --npz_path ../model/data_and_vocab_bt_st_upsample_best.npz \
 --start 200000 --end 400000`

Predictions generated will be saved in an outfile: `predictions_english_monolingual_$(START)_$(END).txt`

### post-process with regex

- `python refine_preds_regex.py --file predictions_english_monolingual_$(START)_$(END).txt`