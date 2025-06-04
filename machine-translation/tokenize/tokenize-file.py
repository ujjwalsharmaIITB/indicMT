import sentencepiece as spm
import argparse
from tqdm import tqdm



def tokenize(input_file, output_file, model_file):
    sp = spm.SentencePieceProcessor()
    sp.Load(model_file)
    with open(input_file, 'r') as src:
        with open(output_file, 'w+' , encoding='utf-8') as tgt:
            for line in tqdm(src, desc="Tokenizing..."):
                tgt.write(' '.join(sp.EncodeAsPieces(line.strip())) + '\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',"-i", type=str, help='input file to tokenize' , required=True)
    parser.add_argument('--output_file',"-o", type=str, help='output file to write tokenized text' , required=True)
    parser.add_argument('--model_file',"-m", type=str, help='sentencepiece model file' , required=True)
    args = parser.parse_args()
    print("starting tokenization...")
    tokenize(args.input_file, args.output_file, args.model_file)
    print("done tokenizing...")