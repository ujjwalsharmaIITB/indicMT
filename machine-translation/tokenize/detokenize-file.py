import sentencepiece as spm
import argparse



def de_tokenise(src_file , tgt_file , model):
    sp = spm.SentencePieceProcessor()
    sp.Load(model)
    with open(src_file, 'r') as src:
        with open(tgt_file, 'w') as tgt:
            for line in src:
                tgt.write(sp.DecodePieces(line.split()) + "\n")
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',"-i", type=str, help='input file to detokenise' , required=True)
    parser.add_argument('--output_file',"-o", type=str, help='output file to write detokenized text' , required=True)
    parser.add_argument('--model_file',"-m", type=str, help='sentencepiece model file' , required=True)
    args = parser.parse_args()
    print("starting detokenisation...")
    de_tokenise(args.input_file, args.output_file, args.model_file)
    print("done detokenizing...")