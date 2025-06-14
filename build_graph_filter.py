import pandas as pd
from Graph_Email import GraphBasedSpamFilter

# load & clean
df = pd.read_csv("emails_dataset.csv")\
        .rename(columns={"Subject":"subject","Body":"body","Labels":"label"})
df["subject"] = df["subject"].fillna("").astype(str)
df["body"]    = df["body"].fillna("").astype(str)
df["label"]   = df["label"].astype(int)

# train & save
gf = GraphBasedSpamFilter(similarity_threshold=0.2)
gf.train_with_dataframe(df)
gf.save("graph_filter.pkl")
print("✅ Rebuilt graph_filter.pkl against Graph_Email.GraphBasedSpamFilter")