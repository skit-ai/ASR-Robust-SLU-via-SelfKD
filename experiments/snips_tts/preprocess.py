import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
os.makedirs("./snips_tts/snips_tts_asr_processed", exist_ok=True)

def sort_hyp(df):
    hyps = df["hypothesis"].to_list()
    scores = df["score"].to_list()
    for i, (h, s) in enumerate(zip(hyps, scores)):
        temp = sorted(list(zip(s, h)), key=lambda x: -x[0])
        temp = tuple([item[1] for item in temp])
        hyps[i] = temp
    df["hypothesis"] = hyps
    return df

encoder = LabelEncoder()
for split in ["train", "valid", "test"]:
    data = pd.read_csv(f"snips_tts/snips_tts_asr/{split}.conf.csv")
    data = data.groupby("transcription").agg(tuple).reset_index()
    data = data.drop(columns=["confusion"])
    labels = pd.read_csv(f"snips_tts/snips_tts_asr/{split}.csv")
    labels2 = pd.read_csv(f"snips_tts/snips_tts/{split}.csv")
    len_before = len(data)

    merged = pd.merge(data, labels, left_on="transcription", right_on="text", how="inner")
    
    # after = set(merged["transcription"].to_list())
    # before = set(data["transcription"].to_list())
    # print(before.symmetric_difference(after))

    merged = sort_hyp(merged)
    merged = merged[["id", "hypothesis", "score", "label"]]
    merged.drop(columns=["score"], inplace=True)
    merged2 = pd.merge(merged, labels2, on="id", how="inner")
    print(merged2.columns, merged.columns)
    assert((merged2["label_x"]==merged2["label_y"]).all())
    merged = merged2
    merged.drop(columns="label_x", inplace=True)
    merged.rename(columns={"label_y": "label", "text": "transcription"}, inplace=True)
    if split == "train":
        encoder = encoder.fit(merged["label"])
    merged["label"] = encoder.transform(merged["label"])
    print("num_classes", len(merged["label"].unique()))
    merged.drop_duplicates(subset=["transcription", "hypothesis"], inplace=True)
    merged.reset_index(drop=True, inplace=True)
    merged.sort_values("id", inplace=True)
    len_after = len(merged)
    print("split", split, "len_before", len_before, "len_after", len_after, "label_len", len(labels))
    merged.to_csv(f"./snips_tts/snips_tts_asr_processed/{split}.csv", index=False)
