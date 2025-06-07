## building vocab
onmt_build_vocab -n_sample -1 -config

## Training 

CUDA_VISIBLE_DEVICES=0,3,4,5 onmt_train -config 


## Inferece
CUDA_VISIBLE_DEVICES=2 onmt_translate -gpu 0 -batch_size 512 -batch_type tokens -beam_size 5 -model  -src  -output 




onmt_release_model --model model.pt --format ctranslate2 --output ct2_model





conda env export --no-build | grep -v "prefix"  > environment.yaml