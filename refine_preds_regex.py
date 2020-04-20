import argparse
import os, re
from tqdm import tqdm

def main(args):
    path = args.file
    file_path_suffix, ext = os.path.splitext(path)
    out_path = file_path_suffix + "_regex" + ext

    print("Processing input file:",path)
    lines = [i.strip() for i in open(path,'r').readlines()]
    # reuse the variable pattern
    pattern = re.compile(r'(\b.+\b)\1\b') # bigram

    out = []
    for line in tqdm(lines):
        while pattern.search(line):
            line = pattern.sub(r'\1', line)
        out.append(line)
    print("Saving to:",out_path,"...")
    with open(out_path,'w') as f:
        for line in tqdm(out):
            print(line,file=f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='get filename to process')
    parser.add_argument('--file', type=str, default="", help='input filename')
    args = parser.parse_args()
    main(args)