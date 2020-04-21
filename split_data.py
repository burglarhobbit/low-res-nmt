from pathlib import Path
import os
from collections import Counter
import numpy as np

np.random.seed(8080)


def read_data(data_path):
    data_path
    os.listdir(data_path)

    with open(data_path / "train.lang1", "r") as f:
        english = f.read()
    len(english.split("\n"))

    with open(data_path / "train.lang2", "r") as f:
        french = f.read()
    len(french.split("\n"))

    return english, french


def split_dataset(data_path, text_data_1, text_data_2, split):
    text_data_1 = np.array(text_data_1.split("\n"))
    text_data_2 = np.array(text_data_2.split("\n"))

    if text_data_1[-1] == "":
        text_data_1 = text_data_1[:-1]

    if text_data_2[-1] == "":
        text_data_2 = text_data_2[:-1]

    idxs = list(range(len(text_data_1)))
    np.random.shuffle(idxs)
    text_data_1 = text_data_1[idxs]
    text_data_2 = text_data_2[idxs]

    train_split = int(len(text_data_1) * split)
    print(train_split, len(text_data_1) - train_split)

    train_text1 = text_data_1[:train_split]
    val_text1 = text_data_1[train_split:]
    train_text2 = text_data_2[:train_split]
    val_text2 = text_data_2[train_split:]

    with open(data_path / 'split_train.lang1', 'w') as f:
        f.write('\n'.join(train_text1))

    with open(data_path / 'split_train.lang2', 'w') as f:
        f.write('\n'.join(train_text2))

    with open(data_path / 'split_val.lang1', 'w') as f:
        f.write('\n'.join(val_text1))

    with open(data_path / 'split_val.lang2', 'w') as f:
        f.write('\n'.join(val_text2))


def main():
    data_path = Path("/content/drive/My Drive/Adv Projects in ML/data")
    english, french = read_data(data_path)
    split_dataset(data_path, english, french, 0.8)

    with open(data_path / "split_train.lang1", "r") as f:
        english_train = f.read()
    print(len(english_train.split("\n")), english_train[:200])

    with open(data_path / "split_train.lang2", "r") as f:
        french_train = f.read()
    print(len(french_train.split("\n")), french_train[:200])

    with open(data_path / "split_val.lang1", "r") as f:
        english_val = f.read()
    print(len(english_val.split("\n")), english_val[:200])

    with open(data_path / "split_val.lang2", "r") as f:
        french_val = f.read()
    print(len(french_val.split("\n")), french_val[:200])


if __name__ == "__main__":
    main()
