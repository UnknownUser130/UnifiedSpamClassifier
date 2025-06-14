import re
import html
import pandas as pd
import nltk
import networkx as nx
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import pickle
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

class GraphBasedSpamFilter:
    def __init__(self, similarity_threshold=0.2,
                 min_df=5,
                 max_df_ratio=0.8,
                 top_k=10):
        self.graph = nx.Graph()
        self.messages = []               # list of (full_text, label)
        self.message_tokens = []         # list of precomputed token sets
        self.token_index = defaultdict(set)
        self.similarity_threshold = similarity_threshold
        self.min_df = min_df
        self.max_df_ratio = max_df_ratio
        self.top_k = top_k

    def preprocess(self, text):
        try:
            text = "" if text is None else str(text)
            text = html.unescape(text.lower())
            if "<" in text and ">" in text:
                text = BeautifulSoup(text, "html.parser").get_text(" ")
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\S+@\S+", " ", text)
            text = re.sub(r"http\S+|www\.\S+", " ", text)
            text = re.sub(r"[^a-z0-9\s]", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            tokens = nltk.word_tokenize(text)
            clean = [
                lemmatizer.lemmatize(w)
                for w in tokens
                if w not in stop_words and len(w) > 1
            ]
            return set(clean)
        except Exception as e:
            print("Preprocess error:", e)
            return set()

    def _meta_tokens(self, meta: dict):
        toks = set()
        dom = meta.get("from_domain")
        if dom:
            toks.add(f"dom_{dom}")
        if meta.get("has_attachment", False):
            toks.add("has_attachment")
        sl = meta.get("subject_length", 0)
        toks.add(f"subj_len_{(sl//50)*50}")
        tc = meta.get("to_count", 0)
        toks.add(f"to_cnt_{min(tc,10)}")
        uc = meta.get("url_count", 0)
        toks.add(f"url_cnt_{min(uc,5)}")
        hr = meta.get("hour")
        if hr is not None and hr >= 0:
            toks.add(f"hour_{hr}")
        wd = meta.get("weekday")
        if wd is not None and wd >= 0:
            toks.add(f"wkday_{wd}")
        return toks

    def cosine_similarity(self, set1, set2):
        inter = len(set1 & set2)
        return inter / ((len(set1)*len(set2))**0.5 + 1e-9)

    def add_message(self, subject, body, label, metadata=None):
        subj_toks = self.preprocess(subject)
        body_toks = self.preprocess(body)
        tokens = subj_toks | body_toks
        if metadata:
            tokens |= self._meta_tokens(metadata)

        idx = len(self.messages)
        self.messages.append((f"{subject} {body}", label))
        self.message_tokens.append(tokens)
        self.graph.add_node(idx, subject=subject, body=body, label=label)

        candidates = {i for t in tokens for i in self.token_index[t]}
        for i in candidates:
            sim = self.cosine_similarity(tokens, self.message_tokens[i])
            if sim > self.similarity_threshold:
                self.graph.add_edge(idx, i, weight=sim)
        for t in tokens:
            self.token_index[t].add(idx)

    def predict_spam_subject_body(self, subject, body, metadata=None):
        subj_toks = self.preprocess(subject)
        body_toks = self.preprocess(body)
        tokens = subj_toks | body_toks
        if metadata:
            tokens |= self._meta_tokens(metadata)

        scores = []
        candidates = {i for t in tokens for i in self.token_index.get(t, [])}
        for i in candidates:
            sim = self.cosine_similarity(tokens, self.message_tokens[i])
            if sim > self.similarity_threshold:
                scores.append(self.messages[i][1])
        # compute confidence = fraction of spam neighbors        
        if not scores:
            return 0.0, 0.0
        conf = sum(scores) / len(scores)
        label = 1.0 if conf > 0.5 else 0.0
        return label, conf

    def train_with_dataframe(self, df):
        # Expect df columns: from, to, date, subject, body, label
        for _, row in df.iterrows():
            meta = {
                "from_domain": row.get("from","").split("@")[-1] if pd.notnull(row.get("from")) and "@" in row.get("from","") else None,
                "to_count": len(str(row.get("to","")).split(",")),
                "subject_length": len(str(row.get("subject","") or "")),
                "has_attachment": False,
                "hour": pd.to_datetime(row.get("date"), errors="coerce").hour if pd.notnull(row.get("date")) else None,
                "weekday": pd.to_datetime(row.get("date"), errors="coerce").weekday() if pd.notnull(row.get("date")) else None,
                "url_count": len(re.findall(r'http[s]?://\S+', str(row.get("body","") or "")))
            }
            self.add_message(row.get("subject",""), row.get("body",""), row.get("label",0), metadata=meta)

        N = len(self.messages)
        df_counts = {t: len(idxs) for t, idxs in self.token_index.items()}
        valid = {t for t, c in df_counts.items()
                 if c >= self.min_df and c <= self.max_df_ratio * N}

        self.token_index = defaultdict(set)
        for i, toks in enumerate(self.message_tokens):
            filt = toks & valid
            self.message_tokens[i] = filt
            for t in filt:
                self.token_index[t].add(i)

        # prune to top_k neighbors
        for u in list(self.graph.nodes()):
            nbrs = [(v, self.graph[u][v]['weight']) for v in self.graph[u]]
            nbrs.sort(key=lambda x: -x[1])
            keep = {v for v, _ in nbrs[:self.top_k]}
            for v in list(self.graph[u]):
                if v not in keep:
                    self.graph.remove_edge(u, v)
                    
    def incremental_update(self, subject, body, predicted_label, metadata=None):
        """
        Incrementally add the newly classified email to the knowledge base.
        predicted_label is 0 (ham) or 1 (spam).
        """
        self.add_message(subject, body, predicted_label, metadata)
    
    def save(self, path: str):
        """Persist this trained filter to disk via pickle."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        """Load a persisted filter from disk via pickle."""
        with open(path, "rb") as f:
            return pickle.load(f)