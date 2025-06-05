import sentencepiece as spm

input_file = '/home/development/ujjwalsharma/indicMT/data/BPCC/final/vocab/vocab.en.hi.bn.gu.kn.mr.ps.ds.ta.ur'
model_prefix = 'vocab/multilingual'
vocab_size = 32000
user_defined_symbols = "<en>,<hi>,<bn>,<gu>,<kn>,<mr>,<pa>,<sa>,<ta>,<ur>"

# Train the SentencePiece model

spm.SentencePieceTrainer.Train(
    f"--input={input_file} "
    f"--model_prefix={model_prefix} "
    f"--vocab_size={vocab_size} "
    f"--user_defined_symbols={user_defined_symbols} "
    "--model_type=unigram "
    "--character_coverage=1 "
    "--bos_id=0 --pad_id=1 --eos_id=2 --unk_id=3 "
    "--train_extremely_large_corpus=true"
)
