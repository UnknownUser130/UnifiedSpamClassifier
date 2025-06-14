import streamlit as st
from streamlit_extras.let_it_rain import rain
import pickle
import pandas as pd
import numpy as np
import time
import re
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from email import message_from_string
from bs4 import BeautifulSoup
import tensorflow as tf
import html
from email.header import decode_header
from Graph_Email import GraphBasedSpamFilter
import os


@st.cache_data(show_spinner=False)
def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

@st.cache_data(show_spinner=False)
def load_corpora(spam_path: str, ham_path: str) -> tuple[str,str]:
    return load_text(spam_path), load_text(ham_path)

@st.cache_data(show_spinner=False)
def load_sms_dataset():
    df = pd.read_csv("super_sms_dataset.csv", encoding="ISO-8859-1")
    df["length"] = df["SMSes"].astype(str).str.len()
    return df

@st.cache_data(show_spinner=False)
def load_email_dataset():
    df = pd.read_csv("emails_dataset.csv")
    df["combined_text"] = df["subject"].fillna("").astype(str) + " " + df["body"].fillna("").astype(str)
    df["length"] = df["combined_text"].str.len()
    return df
def run_app():
    
    # ---------------------- THEME ----------------------
    def custom_theme():
        return """
        <style>
        :root {
            --spam-card-bg: #FFE5E5;
            --spam-card-border: #FF6B6B;
            --ham-card-bg: #E5FFE5;
            --ham-card-border: #4ECDC4;
            --card-shadow: rgba(0, 0, 0, 0.1);
        }
        .result-card {
            padding: 25px;
            margin: 20px 0;
            border-radius: 12px;
            transition: all 0.3s ease;
        }
        .spam-result {
            background-color: var(--spam-card-bg);
            border: 2px solid var(--spam-card-border);
            box-shadow: 0 4px 6px var(--card-shadow);
            color: #000; /* makes text clearly visible in dark mode */
        }
        .ham-result {
            background-color: var(--ham-card-bg);
            border: 2px solid var(--ham-card-border);
            box-shadow: 0 4px 6px var(--card-shadow);
            color: #000; /* makes text clearly visible in dark mode */
        }
        .result-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 15px var(--card-shadow);
        }
        .main-header {
            font-size: 42px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
            transition: color 0.3s ease;
        }
        .sub-header {
            font-size: 18px;
            text-align: center;
            margin-bottom: 30px;
            transition: color 0.3s ease;
        }
        .highlight {
            padding: 2px 4px;
            border-radius: 3px;
            font-weight: 500;
            background-color: rgba(255, 193, 7, 0.3);
        }
        .stButton > button {
            border-radius: 8px;
            padding: 10px 24px;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
        }
        </style>
        """

    st.markdown(custom_theme(), unsafe_allow_html=True)

    # ---------------------- LOAD MODELS & VECTORIZERS ----------------------
    @st.cache_resource
    def load_sms_model():
        vect_sms = pickle.load(open("vectorizer_super.pkl", "rb"))
        model_sms = pickle.load(open("model_bnb_super.pkl", "rb"))
        return vect_sms, model_sms

    @st.cache_resource
    def load_email_model():
        email_model = tf.keras.models.load_model("ann_model.keras")
        subj_vect = pickle.load(open("subject_vectorizer.pkl", "rb"))
        body_vect = pickle.load(open("body_vectorizer.pkl", "rb"))
        return email_model, subj_vect, body_vect

    @st.cache_resource
    def init_graph_filter(path="graph_filter_meta.pkl", threshold=0.2):
        if os.path.exists(path):
            return GraphBasedSpamFilter.load(path)

        df = (pd.read_csv("emails_dataset.csv")
                .rename(columns={"Subject":"subject","Body":"body","Labels":"label"}))
        df["subject"] = df["subject"].fillna("").astype(str)
        df["body"]    = df["body"].fillna("").astype(str)
        # map "ham"/"spam" → 0/1
        df["label"] = (df["label"]
                        .map({"ham":0,"spam":1})
                        .fillna(0)
                        .astype(int))

        gf = GraphBasedSpamFilter(similarity_threshold=threshold)
        gf.train_with_dataframe(df)
        gf.save(path)
        return gf

    vect_sms, model_sms = load_sms_model()
    model_email, vect_email_subj, vect_email_body = load_email_model()

    # ---------------------- HELPER FUNCTIONS ----------------------
    def highlight_text(text, trigger_words):
        """Wrap each trigger word in a highlight span."""
        highlight_style = '<span class="highlight">{}</span>'
        if not trigger_words:
            return text
        pattern = r'\b(' + '|'.join(re.escape(w) for w in trigger_words) + r')\b'
        return re.sub(pattern, lambda m: highlight_style.format(m.group()), text, flags=re.IGNORECASE)

    def get_spam_triggers(text, vectorizer):
        """Find active features in text."""
        # vectorizer must have get_feature_names_out()
        feat_names = vectorizer.get_feature_names_out()
        vec = vectorizer.transform([text]).toarray()
        return [feat_names[i] for i, val in enumerate(vec[0]) if val > 0]

    def show_result(is_spam, conf, highlighted, hide_conf=False):
        """Render spam/ham result visually."""
        if is_spam:
            rain(emoji="⚠️", font_size=54, falling_speed=5, animation_length="1s")
            st.markdown(
    f"""<div class="spam-result result-card">
    <h3 style="color: #e74c3c;">🚨 Spam Detected!</h3>
    {"<p>Model Confidence: {:.2f}</p>".format(conf) if not hide_conf else ""}
    <p>Highlighted Message:</p>
    <p>{highlighted}</p>
    </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.balloons()
            st.markdown(
    f"""<div class="ham-result result-card">
    <h3 style="color: #2ecc71;">✅ Not Spam!</h3>
    {"<p>Model Confidence: {:.2f}</p>".format(conf) if not hide_conf else ""}
    <p>Highlighted Message:</p>
    <p>{highlighted}</p>
    </div>""",
                unsafe_allow_html=True,
            )

    def process_file_sms(file_obj):
        """Batch classify SMS with 'v2' column."""
        if file_obj.name.endswith(".csv"):
            df = pd.read_csv(file_obj, encoding="ISO-8859-1")
        else:
            df = pd.read_json(file_obj, orient="records")
        df["Prediction"] = df["v2"].apply(
            lambda x: "Spam" if model_sms.predict(vect_sms.transform([x]))[0] == 1 else "Not Spam"
        )
        return df

    def process_file_email(file_obj):
        """Batch classify Email with 'subject' and 'body' columns."""
        if file_obj.name.endswith(".csv"):
            df = pd.read_csv(file_obj)
        else:
            df = pd.read_json(file_obj, orient="records")
        subj_vec = vect_email_subj.transform(df["Subject"].fillna("")).toarray()
        body_vec = vect_email_body.transform(df["Body"].fillna("")).toarray()
        X = np.hstack((subj_vec, body_vec))
        out = model_email.predict(X)
        preds = []
        for row in out:
            if len(row) == 1:
                # Single output => spam if >= 0.5
                preds.append("Spam" if row[0] >= 0.5 else "Not Spam")
            else:
                # Two-output => [prob0, prob1]
                preds.append("Spam" if row[1] > row[0] else "Not Spam")
        df["Prediction"] = preds
        return df

    def extract_email_content(raw_email):
        """Extract and clean subject and body from raw email text."""
        parsed = message_from_string(raw_email)
        raw_subject = parsed["Subject"] or ""
        decoded_fragments = decode_header(raw_subject)
        subject = ""
        for fragment, encoding in decoded_fragments:
            if isinstance(fragment, bytes):
                subject += fragment.decode(encoding or "utf-8", errors="replace")
            else:
                subject += fragment

        body = ""

        if parsed.is_multipart():
            for part in parsed.walk():
                ct = part.get_content_type()
                payload = part.get_payload(decode=True)
                if payload is None:
                    continue
                try:
                    payload = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
                except Exception:
                    payload = str(payload)

                # Unescape HTML entities (&nbsp;, &amp;, etc.)
                payload = html.unescape(payload)

                if ct == "text/plain":
                    body += payload
                elif ct == "text/html":
                    soup = BeautifulSoup(payload, "html.parser")
                    # Remove script/style tags and comments
                    for tag in soup(["script", "style"]):
                        tag.decompose()
                    # Keep only visible text
                    text = soup.get_text(separator=" ")
                    body += text
        else:
            payload = parsed.get_payload(decode=True)
            if payload is not None:
                try:
                    payload = payload.decode(parsed.get_content_charset() or "utf-8", errors="replace")
                except Exception:
                    payload = str(payload)
                payload = html.unescape(payload)
                body = payload

        # Combine subject and body for more context
        full_text = f"{subject} {body}"

        # Remove angle-bracketed metadata (like <abc@xyz.com> or HTML tags)
        full_text = re.sub(r"<[^>]+>", " ", full_text)
        
        # Replace URLs with placeholder
        full_text = re.sub(r"http\S+|www\S+", " [URL] ", full_text)
        
        # Remove excessive whitespace and non-printable characters
        full_text = re.sub(r"\s+", " ", full_text).strip()
        
        return subject, full_text
    
    def extract_email_metadata(raw_email: str) -> dict:
        """Parse headers and compute the same meta-features used in training."""
        parsed = message_from_string(raw_email)
        # from_domain
        frm = parsed.get("From", "")
        domain = frm.split("@")[-1] if "@" in frm else None
        # to_count
        tos = parsed.get_all("To", []) or []
        # date → hour, weekday
        date_hdr = parsed.get("Date", "")
        ts = pd.to_datetime(date_hdr, errors="coerce")
        return {
            "from_domain": domain,
            "to_count": len(tos),
            "subject_length": len(parsed.get("Subject") or ""),
            "has_attachment": any(part.get_filename() for part in parsed.walk() if part.get_filename()),
            "hour": ts.hour if not pd.isna(ts) else None,
            "weekday": ts.weekday() if not pd.isna(ts) else None,
            "url_count": len(re.findall(r'http[s]?://\S+', raw_email))
        }
    
    # Title & header
    st.markdown('<div class="main-header">📨 Unified SMS & Email Spam Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Uses Naive Bayes for SMS and Graph filter & ANN for Emails.</div>', unsafe_allow_html=True)

    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📩 Classifier", "📊 Insights", "ℹ️ About"])
    data_type = st.sidebar.selectbox("Choose Data Type", ["SMS", "Email"])

    # ---------------------- TAB 1: CLASSIFIER ----------------------
    with tab1:
        st.markdown("<div class='tabs-header'>Single or Batch Classification</div>", unsafe_allow_html=True)
        single_msg_tab, batch_tab = st.tabs(["Single Message", "Batch Processing"])

        # Single
        with single_msg_tab:
            if data_type == "SMS":
                input_text = st.text_area("Type your SMS:", placeholder="Type your SMS here...", height=150)
                # load SMS sample from external txt
                sample_sms_data = load_text("data/sample_sms.txt")
                
                st.download_button(
                    label="Download Sample SMS Text File",
                    data=sample_sms_data,
                    file_name="sample_sms.txt",
                    mime="text/plain"
                )
            else:
                raw_email = st.text_area("Paste raw email content:", placeholder="Paste entire raw email here...", height=250)
                
                # load Email sample from external txt
                sample_email_data = load_text("data/sample_email.txt")

                st.download_button(
                    label="Download Sample Email Text File",
                    data=sample_email_data,
                    file_name="sample_email.txt",
                    mime="text/plain"
                )
            if st.button("🔮 Classify Message"):
                with st.spinner("Analyzing..."):
                    time.sleep(1)
                    if data_type == "SMS":
                        if not input_text.strip():
                            st.warning("⚠️ Please enter a valid SMS.")
                            return
                        x_vec = vect_sms.transform([input_text]).toarray()
                        pred = model_sms.predict(x_vec)[0]
                        conf = model_sms.predict_proba(x_vec)[0].max()
                        triggers = get_spam_triggers(input_text, vect_sms)
                        highlighted = highlight_text(input_text, triggers)
                        show_result(pred, conf, highlighted)
                    else:
                        if not raw_email.strip():
                            st.warning("⚠️ Please enter a valid email.")
                            return

                        # 1) Extract subject + body
                        subj, body = extract_email_content(raw_email)

                        # 2) Let user pick which email model to use
                        
                        model_option = st.radio(
                            "Select Email Model:",
                            ["ANN", "Graph-based"],
                            index=0,
                            key="email_model_option"
                        )
                        if model_option == "Graph-based":
                            # Graph‐based prediction with metadata
                            graph_filter = init_graph_filter(threshold=0.2)
                            meta = extract_email_metadata(raw_email)
                            is_spam, conf = graph_filter.predict_spam_subject_body(subj, body, metadata=meta)
                        
                        else:
                            #ANN logic
                            s_vec = vect_email_subj.transform([subj]).toarray()
                            b_vec = vect_email_body.transform([body]).toarray()
                            X_eb  = np.hstack((s_vec, b_vec))
                            out   = model_email.predict(X_eb)[0]
                            if len(out) == 1:
                                spam_score = float(out[0])
                                is_spam    = (spam_score >= 0.5)
                                conf       = spam_score if is_spam else (1.0 - spam_score)
                            else:
                                is_spam = (out[1] > out[0])
                                conf    = float(max(out[0], out[1]))

                        # 3) Continue with highlighting & display
                        triggers    = get_spam_triggers(body, vect_email_body)
                        highlighted = highlight_text(body, triggers)
                        show_result(is_spam, conf, highlighted)

                        pred = is_spam
                        input_text = raw_email
            
            if "user_results" not in st.session_state:
                st.session_state["user_results"] = {}

            username = st.session_state.get("current_user", st.session_state.get("login_username", ""))

            
        # Batch
        with batch_tab:
            up_caption = "Upload a file (CSV/JSON) for SMS" if data_type == "SMS" else "Upload a file (CSV/JSON) for Email"
            file_up = st.file_uploader(up_caption, type=["csv", "json"])
            if file_up:
                with st.spinner("Processing..."):
                    if data_type == "SMS":
                        df = process_file_sms(file_up)
                    else:
                        df = process_file_email(file_up)
                st.dataframe(df)
                st.download_button("Download CSV", df.to_csv(index=False), "results.csv", "text/csv")

    # ---------------------- TAB 2: INSIGHTS (SMS data and Email Data) ----------------------
    with tab2:
        if data_type == "SMS":
            st.markdown('<div class="tabs-header">📊 Model Insights</div>', unsafe_allow_html=True)

            #SMS dataset stats
            total_messages = 67008
            spam_count = 26178
            ham_count = total_messages - spam_count

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Messages", f"{total_messages:,}")
            col2.metric("Spam Messages", f"{spam_count:,}", f"{(spam_count/total_messages)*100:.1f}%")
            col3.metric("Ham Messages", f"{ham_count:,}", f"{(ham_count/total_messages)*100:.1f}%")

            st.subheader("📈 Model Performance")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Accuracy", "98.05%")
            c2.metric("Precision", "98.11%")
            c3.metric("Recall", "96.88%")
            c4.metric("F1", "97.49%")
            c5.metric("Specificity", "98.80%")

            # Confidence threshold
            st.subheader("🎯 Confidence Threshold")
            threshold = st.slider("Adjust classification threshold", 0.0, 1.0, 0.5, 0.05)

            # Visualizations
            viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📊 Distribution", "🔤 Word Cloud", "📉 Confusion Matrix"])

            with viz_tab1:
                # distribution of message lengths from dataset
                try:
                    df = load_sms_dataset()
                    fig = go.Figure()

                    spam_len = df[df["Labels"] == 1]["length"]
                    ham_len = df[df["Labels"] == 0]["length"]

                    fig.add_trace(go.Histogram(x=spam_len, name="Spam", nbinsx=30, marker_color="#FF6B6B", opacity=0.7))
                    fig.add_trace(go.Histogram(x=ham_len, name="Ham", nbinsx=30, marker_color="#4ECDC4", opacity=0.7))

                    fig.update_layout(
                        title="Message Length Distribution",
                        xaxis_title="Length (chars)",
                        yaxis_title="Count",
                        barmode="overlay",
                        bargap=0.1,
                    )
                    fig.update_traces(hoverinfo="x+y")

                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning("Unable to render length distribution (dataset missing).")

            with viz_tab2:
                # Word clouds

                spam_text, ham_text = load_corpora("spam_corpus.txt","ham_corpus.txt")
                col_spam, col_ham = st.columns(2)
                try:
                    with col_spam:
                        st.subheader("Spam Words")
                        wc_spam = WordCloud(width=800, height=400, background_color='white').generate(spam_text)
                        fig_s = plt.figure(figsize=(10,5))
                        plt.imshow(wc_spam.to_array())
                        plt.axis('off')
                        st.pyplot(fig_s)
                        plt.close(fig_s)

                    with col_ham:
                        st.subheader("Ham Words")
                        wc_ham = WordCloud(width=800, height=400, background_color='white').generate(ham_text)
                        fig_h = plt.figure(figsize=(10,5))
                        plt.imshow(wc_ham.to_array())
                        plt.axis('off')
                        st.pyplot(fig_h)
                        plt.close(fig_h)

                except Exception as e:
                    st.error("Could not generate word clouds. Check your 'spam_corpus.txt' and 'ham_corpus.txt'.")

            with viz_tab3:
                # confusion matrix
                conf_matrix = np.array([[12112, 146],
                                        [247, 7598]])
                fig_c = px.imshow(conf_matrix, labels=dict(x="Predicted", y="Actual"),
                                x=["Spam","Ham"], y=["Spam","Ham"])
                fig_c.update_xaxes(side="top")
                st.plotly_chart(fig_c, use_container_width=True)

            # Top spam triggers 
            st.subheader("🚩 Top Spam Triggers")
            triggers_data = pd.DataFrame({
                "Word": ["send", "free", "offer", "win", "click", "now", "urgent"],
                "Frequency": [8000, 7200, 5800, 5500, 4020, 3700, 3500]
            })
            st.bar_chart(triggers_data.set_index("Word"))
        else:
            st.markdown('<div class="tabs-header">📊 Model Insights</div>', unsafe_allow_html=True)

            # stats from Email dataset
            total_messages = 17754
            spam_count = 5356
            ham_count = total_messages - spam_count

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Messages", f"{total_messages:,}")
            col2.metric("Spam Messages", f"{spam_count:,}", f"{(spam_count/total_messages)*100:.1f}%")
            col3.metric("Ham Messages", f"{ham_count:,}", f"{(ham_count/total_messages)*100:.1f}%")

            st.subheader("📈 Model Performance")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Accuracy", "95.12%")
            c2.metric("Precision", "92.81%")
            c3.metric("Recall", "92.16%")
            c4.metric("F1", "96.7%")
            c5.metric("Specificity", "91.45%")

            # Confidence threshold
            st.subheader("🎯 Confidence Threshold")
            threshold = st.slider("Adjust classification threshold", 0.0, 1.0, 0.5, 0.05)

            # Visualizations
            viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📊 Distribution", "🔤 Word Cloud", "📉 Confusion Matrix"])

            with viz_tab1:
                # distribution of message lengths from a sample dataset
                try:
                    df = load_email_dataset()
                    df["combined_text"] = df["subject"].fillna("").astype(str) + " " + df["body"].fillna("").astype(str)
                    df["length"] = df["combined_text"].str.len()

                    fig = go.Figure()

                    spam_len = df[df["label"] == 1]["length"]
                    ham_len = df[df["label"] == 0]["length"]

                    fig.add_trace(go.Histogram(x=spam_len, name="Spam", nbinsx=30, marker_color="#FF6B6B", opacity=0.7))
                    fig.add_trace(go.Histogram(x=ham_len, name="Ham", nbinsx=30, marker_color="#4ECDC4", opacity=0.7))

                    fig.update_layout(
                        title="Email (Subject + Body) Length Distribution",
                        xaxis_title="Length (chars)",
                        yaxis_title="Count",
                        barmode="overlay",
                        bargap=0.1,
                    )
                    fig.update_traces(hoverinfo="x+y")

                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning("Unable to render length distribution (dataset missing).")

            with viz_tab2:
                # Word clouds

                spam_text, ham_text = load_corpora("spam_corpus_email.txt","ham_corpus_email.txt")
                col_spam, col_ham = st.columns(2)
                try:
                    with col_spam:
                        st.subheader("Spam Words")
                        wc_spam = WordCloud(width=800, height=400, background_color='white').generate(spam_text)
                        fig_s = plt.figure(figsize=(10,5))
                        plt.imshow(wc_spam.to_array())
                        plt.axis('off')
                        st.pyplot(fig_s)
                        plt.close(fig_s)

                    with col_ham:
                        st.subheader("Ham Words")
                        wc_ham = WordCloud(width=800, height=400, background_color='white').generate(ham_text)
                        fig_h = plt.figure(figsize=(10,5))
                        plt.imshow(wc_ham.to_array())
                        plt.axis('off')
                        st.pyplot(fig_h)
                        plt.close(fig_h)

                except Exception as e:
                    st.error("Could not generate word clouds. Check your 'spam_corpus.txt' and 'ham_corpus.txt'.")

            with viz_tab3:
                # confusion matrix
                conf_matrix = np.array([[2280, 18],
                                        [25, 1028]])
                fig_c = px.imshow(conf_matrix, labels=dict(x="Predicted", y="Actual"),
                                x=["Spam","Ham"], y=["Spam","Ham"])
                fig_c.update_xaxes(side="top")
                st.plotly_chart(fig_c, use_container_width=True)

            # Top spam triggers
            st.subheader("🚩 Top Spam Triggers")
            triggers_data = pd.DataFrame({
                "Word": ["email", "company", "free", "please", "business", "get", "information"],
                "Frequency": [6325, 3787, 3747, 3075, 3063, 3032, 2972]
            })
            st.bar_chart(triggers_data.set_index("Word"))

    # ---------------------- TAB 3: ABOUT ----------------------
    with tab3:
        st.markdown('<div class="tabs-header">ℹ️ About This Project</div>', unsafe_allow_html=True)
        st.write("This **unified** spam classifier handles SMS and Email. "
                 "It includes many features like word clouds, confusion matrices, and distribution plots.")

        st.image("https://s38924.pcdn.co/wp-content/uploads/2019/12/6-Tips-to-Reduce-Spam-Form-Entries.jpg", use_column_width=True)

        st.markdown("""
        **Technologies used**:  
        • **SMS**: scikit-learn, NLTK, Pandas, NumPy  
        • **Email**: Keras/TensorFlow, BeautifulSoup, HTML Parser
        • **Visualization**: Matplotlib, Plotly, Streamlit
        • WordCloud, Plotly, Streamlit  
        
        **Note**: This is a demo for our Project.
        """)

    st.markdown("---")
    st.markdown("Unified Spam Detection – Final Year Project-II")
