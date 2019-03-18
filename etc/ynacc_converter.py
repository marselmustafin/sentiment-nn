import pandas as pd

NAMES = ["sdid", "commentindex", "headline", "url", "guid", "commentid",
          "timestamp", "thumbs-up", "thumbs-down", "text", "parentid",
          "constructiveclass", "sd_agreement", "sd_type", "sentiment", "tone",
          "commentagreement", "topic", "intendedaudience", "persuasiveness"]
SENTIMENTS = ["negative", "neutral", "positive"]

YNACC_EXPERT_CORPUS = "data/ydata-ynacc-v1_0_expert_annotations.tsv"
CONVERTED_CORPUS = "data/train/ydata-ynacc-v1_0_expert_annotations_filt.tsv"

data = pd.read_csv(YNACC_EXPERT_CORPUS, header=None, sep="\t", names=NAMES)
data = data[data.sentiment.isin(SENTIMENTS)]
data = data[data.apply(lambda x: len(x.text.split(" ")) < 90, axis=1)]
data.to_csv(path_or_buf=CONVERTED_CORPUS, sep="\t", header=None,
                index=False, columns=["commentid", "sentiment", "text"])
