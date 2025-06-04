# %%
import pandas as pd
import numpy as np

def load_data(src_file, tgt_file):
    src_lines = [x.strip() for x in open(src_file, 'r', encoding='utf-8')]
    tgt_lines = [x.strip() for x in open(tgt_file, 'r', encoding='utf-8')]

    df = pd.DataFrame({'src': src_lines, 'tgt': tgt_lines})
    df = df.dropna()

    dev_df = df.sample(n=1001, random_state=42)
    df = df.drop(dev_df.index)

    return df, dev_df



from tqdm import tqdm

def save_df_to_file(df, base_path, split):
    with open(base_path + f"/{split}.multilingual.src", 'w', encoding='utf-8') as src_file, \
         open(base_path + f"/{split}.multilingual.tgt", 'w', encoding='utf-8') as tgt_file:
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Saving to files"):
            src_file.write(str(row['src']) + "\n")
            tgt_file.write(str(row['tgt']) + "\n")

# english_bengali

english_bengali_train, english_bengali_dev = load_data("/home/development/ujjwalsharma/indicMT/data/BPCC/final/en-bn/en-bn.src",
                            "/home/development/ujjwalsharma/indicMT/data/BPCC/final/en-bn/en-bn.tgt")                              


print(english_bengali_train.count())

# english_gujarati

english_gujarati_train, english_gujarati_dev = load_data("/home/development/ujjwalsharma/indicMT/data/BPCC/final/en-gu/en-gu.src",
                            "/home/development/ujjwalsharma/indicMT/data/BPCC/final/en-gu/en-gu.tgt")

print(english_gujarati_train.count())

# english_hindi
english_hindi_train, english_hindi_dev = load_data("/home/development/ujjwalsharma/indicMT/data/BPCC/final/en-hi/en-hi.src",
                            "/home/development/ujjwalsharma/indicMT/data/BPCC/final/en-hi/en-hi.tgt")
print(english_hindi_train.count())

# english_kannada
english_kannada_train, english_kannada_dev  = load_data("/home/development/ujjwalsharma/indicMT/data/BPCC/final/en-kn/en-kn.src",
                            "/home/development/ujjwalsharma/indicMT/data/BPCC/final/en-kn/en-kn.tgt")

print(english_kannada_train.count())

# english_marathi

english_marathi_train, english_marathi_dev = load_data("/home/development/ujjwalsharma/indicMT/data/BPCC/final/en-mr/en-mr.src",
                            "/home/development/ujjwalsharma/indicMT/data/BPCC/final/en-mr/en-mr.tgt")
print(english_marathi_train.count())

# english_punjabi

english_punjabi_train, english_punjabi_dev = load_data("/home/development/ujjwalsharma/indicMT/data/BPCC/final/en-pa/en-pa.src",
                            "/home/development/ujjwalsharma/indicMT/data/BPCC/final/en-pa/en-pa.tgt")

print(english_punjabi_train.count())

# english_sanskrit

english_sanskrit_train, english_sanskrit_dev = load_data("/home/development/ujjwalsharma/indicMT/data/BPCC/final/en-sa/en-sa.src",
                            "/home/development/ujjwalsharma/indicMT/data/BPCC/final/en-sa/en-sa.tgt")
print(english_sanskrit_train.count())

# english_tamil
english_tamil_train, english_tamil_dev = load_data("/home/development/ujjwalsharma/indicMT/data/BPCC/final/en-ta/en-ta.src",
                            "/home/development/ujjwalsharma/indicMT/data/BPCC/final/en-ta/en-ta.tgt")
print(english_tamil_train.count())

# english_urdu
english_urdu_train, english_urdu_dev = load_data("/home/development/ujjwalsharma/indicMT/data/BPCC/final/en-ur/en-ur.src",
                            "/home/development/ujjwalsharma/indicMT/data/BPCC/final/en-ur/en-ur.tgt")

print(english_urdu_train.count())



final_dev_dataset = pd.concat([
    english_bengali_dev,
    english_gujarati_dev,
    english_hindi_dev,
    english_kannada_dev,
    english_marathi_dev,
    english_punjabi_dev,
    english_sanskrit_dev,
    english_tamil_dev,
    english_urdu_dev
], ignore_index=True).drop_duplicates().sample(frac=1, random_state=42).reset_index(drop=True)


save_df_to_file(final_dev_dataset, "./", "dev")


final_train_dataset = pd.concat([
    english_bengali_train,
    english_gujarati_train,
    english_hindi_train,
    english_kannada_train,
    english_marathi_train,
    english_punjabi_train,
    english_sanskrit_train,
    english_tamil_train,
    english_urdu_train
], ignore_index=True).drop_duplicates().sample(frac=1, random_state=42).reset_index(drop=True)



save_df_to_file(final_train_dataset, "./", "train")