import sentencepiece as spm
import argparse


def train_spm(input_file, model_prefix, vocab_size):
    spm.SentencePieceTrainer.Train(
        f"--input={input_file} --model_prefix={model_prefix} --vocab_size={vocab_size} --model_type=unigram --character_coverage=1 --bos_id=0 --pad_id=1 --eos_id=2 --unk_id=3 --train_extremely_large_corpus=true"
    )

if __name__ == "__main__":

    argsparser = argparse.ArgumentParser()

    argsparser.add_argument('--input_file', '-i', type=str, required=True, help='input file path')
    argsparser.add_argument('--model_prefix', '-m', type=str, required=True, help='model prefix')
    argsparser.add_argument('--vocab_size', '-v', type=int, default=30000, help='vocab size')

    args = argsparser.parse_args()

    print("starting training")
    train_spm(args.input_file, args.model_prefix, args.vocab_size)
    print("training completed")
