import streamlit as st
from streamlit_extras.let_it_rain import rain
import pickle
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from wordcloud import WordCloud
import time
import re

# --------------------------- Streamlit Configuration ---------------------------
st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📨",
    layout="wide",
    initial_sidebar_state="expanded",
    )
def run_app():

    if 'theme' not in st.session_state:
        st.session_state.theme = 'light'


    def custom_theme():
        """Handle and return appropriate CSS"""
        # Add theme toggle to sidebar
    # Custom CSS for Styling
        return f"""
        <style>
        :root {{
            --spam-card-bg: #FFE5E5;
            --spam-card-border: #FF6B6B;
            --ham-card-bg: #E5FFE5;
            --ham-card-border: #4ECDC4;
            --card-shadow: rgba(0, 0, 0, 0.1);
        }}
        .result-card {{
            padding: 25px;
            margin: 20px 0;
            border-radius: 12px;
            transition: all 0.3s ease;
        }}

        .spam-result {{
            background-color: var(--spam-card-bg);
            border: 2px solid var(--spam-card-border);
            box-shadow: 0 4px 6px var(--card-shadow);
        }}

        .ham-result {{
            background-color: var(--ham-card-bg);
            border: 2px solid var(--ham-card-border);
            box-shadow: 0 4px 6px var(--card-shadow);
        }}

        .result-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 15px var(--card-shadow);
        }}
        .main-header {{
            font-size: 42px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
            transition: color 0.3s ease;
        }}

        .sub-header {{
            font-size: 18px;
            text-align: center;
            margin-bottom: 30px;
            transition: color 0.3s ease;
        }}

        .highlight {{
            padding: 2px 4px;
            border-radius: 3px;
            font-weight: 500;
            background-color: rgba(255, 193, 7, 0.3);
        }}

        /* Button Styles */
        .stButton > button {{
            background-color: var(--accent-color);
            color: white;
            border-radius: 8px;
            padding: 10px 24px;
            transition: all 0.3s ease;
        }}

        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }}
        body {{
            background-color: var(--background);
            color: var(--text);
            transition: all 0.3s ease;
        }}
        </style>
    """
    st.markdown(custom_theme(), unsafe_allow_html=True)

    # --------------------------- Load Model and Vectorizer ---------------------------
    @st.cache_resource
    def load_resources():
        vectorizer = pickle.load(open("vectorizer_super.pkl", "rb"))
        model = pickle.load(open("model_bnb_super.pkl", "rb"))
        return vectorizer, model

    vectorizer, model = load_resources()

    # --------------------------- Helper Functions ---------------------------

    def highlight_text(text, trigger_words):
        """Highlight spam trigger words in text."""
        highlight_style = '<span class="highlight">{}</span>'
        pattern = r'\b(' + '|'.join(re.escape(word) for word in trigger_words) + r')\b'
        return re.sub(pattern, lambda match: highlight_style.format(match.group()), text, flags=re.IGNORECASE)

    def get_spam_triggers(text, vectorizer):
        """Extract high-weight spam words from text."""
        feature_names = vectorizer.get_feature_names_out()
        vector = vectorizer.transform([text]).toarray()
        spam_words = [feature_names[i] for i, val in enumerate(vector[0]) if val > 0]
        return spam_words

    def process_file(file):
        """Process uploaded file (CSV or TXT) and return a DataFrame."""
        if file.name.endswith(".csv"):
            df = pd.read_csv(file, encoding="ISO-8859-1")
        elif file.name.endswith(".txt"):
            with open(file, "r") as f:
                data = f.readlines()
            df = pd.DataFrame(data, columns=["v2"])
        df["Prediction"] = df["v2"].apply(
            lambda x: "Spam" if model.predict(vectorizer.transform([x]))[0] == 1 else "Not Spam"
        )
        return df

    def import_text_corpora():
            """Import raw text files for spam and ham corpora"""
            try:
                # Import spam text
                with open('spam_corpus.txt', 'r', encoding='utf-8') as f:
                    spam_text = f.read()
                
                # Import ham text    
                with open('ham_corpus.txt', 'r', encoding='utf-8') as f:
                    ham_text = f.read()
                
                return spam_text, ham_text
            except FileNotFoundError:
                st.error("Corpus files not found. Please check file paths.")
                return "", ""
            except Exception as e:
                st.error(f"Error importing corpora: {e}")
                return "", ""

    # --------------------------- UI Design ---------------------------

    # Header
    st.markdown('<div class="main-header">📨 Spam Classifier App</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Leverage AI to detect Spam messages with high accuracy!</div>', unsafe_allow_html=True)

    # Tabs for Different Features
    tab1, tab2, tab3 = st.tabs(["📩 Classifier", "📊 Insights", "ℹ️ About"])

    # ---------------------- Tab 1: Classifier ----------------------
    with tab1:
        # Create subtabs for single and batch classification
        single_tab, batch_tab = st.tabs(["Single Message", "Batch Processing"])
        st.markdown('<div class="tabs-header">📩 Message Classification</div>', unsafe_allow_html=True)
        
        # ---------------------- Single Message Classification ----------------------
        with single_tab:
        # Input and Predict Section
            input_text = st.text_area("Type your message:", placeholder="Type here...", height=150)

            if st.button("🔮 Classify Message"):
                if input_text.strip() == "":
                    st.warning("⚠️ Please enter a valid message to classify.")
                else:
                    with st.spinner("Analyzing your message..."):
                        time.sleep(1)  # Simulated loading time
                        # Preprocessing and Prediction
                        input_vectorized = vectorizer.transform([input_text]).toarray()
                        prediction = model.predict(input_vectorized)[0]
                        confidence = model.predict_proba(input_vectorized)[0].max()
                        trigger_words = get_spam_triggers(input_text, vectorizer)
                        highlight_text = highlight_text(input_text, trigger_words)

                # Display Results
                
                    if prediction == 1:
                        rain(emoji="⚠️",font_size=54,falling_speed=5,animation_length="1s")
                        st.markdown(
                            f"""
                            <div class="Spam-card">
                                <h3 style="color: #e74c3c;">🚨 Spam Detected!</h3>
                                <p>Model Confidence: <span class='confidence'>{confidence:.2f}</span></p>
                                <p>Highlighted Message:</p>
                                <p>{highlight_text}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.balloons()
                        st.markdown(
                            f"""
                            <div class="Ham-card">
                                <h3 style="color: #2ecc71;">✅ Not Spam!</h3>
                                <p>Model Confidence: <span class='confidence'>{confidence:.2f}</span></p>
                                <p>Highlighted Message:</p>
                                <p>{highlight_text}</p>
                                </div>
                            """,
                            unsafe_allow_html=True,
                        )                                
        with batch_tab:
            # File Upload
            uploaded_file = st.file_uploader("Upload a file (CSV/TXT) to classify multiple messages:", type=["csv", "txt"])
            if uploaded_file:
                results_df = process_file(uploaded_file)
                st.dataframe(results_df)
                st.download_button(
                    "Download Results as CSV",
                    results_df.to_csv(index=False),
                    "classified_messages.csv",
                    "text/csv",
                )
            
        text_data ="""
        Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's.

        SMS. ac Sptv: The New Jersey Devils and the Detroit Red Wings play Ice Hockey. Correct or Incorrect? End? Reply END SPTV.

        07732584351 - Rodger Burns - MSG = We tried to call you re your reply to our sms for a free nokia mobile + free camcorder. Please call now 08000930705 for delivery tomorrow.

        Urgent UR awarded a complimentary trip to EuroDisinc Trav, Aco&Entry41 Or å£1000. To claim txt DIS to 87121 18+6*å£1.50(moreFrmMob. ShrAcomOrSglSuplt)10, LS1 3AJ			

        Please call our customer service representative on 0800 169 6031 between 10am-9pm as you have WON a guaranteed å£1000 cash or £5000 prize!

        Your free ringtone is waiting to be collected. Simply text the password MIX  to 85069 to verify. Get Usher and Britney. FM  PO Box 5249	 MK17 92H. 450Ppw 16"	
                
        Sunshine Quiz Wkly Q! Win a top Sony DVD player if u know which country the Algarve is in? Txt ansr to 82277. å£1.50 SP:Tyrone			

        You'll not rcv any more msgs from the chat svc. For FREE Hardcore services text GO to: 69988 If u get nothing u must Age Verify with yr network & try again	

        Customer service annoncement. You have a New Years delivery waiting for you. Please call 07046744435 now to arrange delivery.
                
        URGENT! We are trying to contact you. Last weekends draw shows that you have won a å£900 prize GUARANTEED. Call 09061701939. Claim code S89. Valid 12hrs only			
            
        """

    # Create a download button for the text file
        st.download_button(
        label="Download Sample Spam Text File",
        data=text_data,
        file_name="sample.txt",
        mime="text/plain"
    )        

    # ---------------------- Tab 2: Insights ----------------------
    with tab2:

        st.markdown('<div class="tabs-header">📊 Model Insights</div>', unsafe_allow_html=True)
        total_messages = 67008
        spam_count = 26178
        non_spam_count = 40830
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Messages", f"{total_messages:,}", "100%")
        with col2:
            st.metric("Spam Messages", f"{spam_count:,}", f"{(spam_count/total_messages)*100:.1f}%")
        with col3:
            st.metric("Ham Messages", f"{non_spam_count:,}", f"{(non_spam_count/total_messages)*100:.1f}%")

        # Model Performance Metrics
        st.subheader("📈 Model Performance")
        metrics_col1, metrics_col2, metrics_col3, metrics_col4, metrics_col5 = st.columns(5)
        with metrics_col1:
            st.metric("Accuracy", "98.05%")
        with metrics_col2:
            st.metric("Precision", "98.11%")
        with metrics_col3:
            st.metric("Recall", "96.88%")
        with metrics_col4:
            st.metric("F1 Score", "97.49%")
        with metrics_col5:
            st.metric("Specificity", "98.80%")

        # Add confidence threshold slider
        st.subheader("🎯 Confidence Threshold")
        threshold = st.slider("Adjust classification threshold", 0.0, 1.0, 0.5, 0.05)

        # Create tabs for different visualizations
        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📊 Distribution", "🔤 Word Cloud", "📉 Confusion Matrix"])
        
        with viz_tab1:
            
            # Load dataset
            df = pd.read_csv('super_sms_dataset.csv', encoding="ISO-8859-1")
        
            # Calculate message lengths
            df['length'] = df['SMSes'].astype(str).str.len()
        
            # Create distribution plot
            fig_dist = go.Figure()
        
            # Add spam trace
            spam_lengths = df[df['Labels'] == 1]['length']
            fig_dist.add_trace(go.Histogram(
                x=spam_lengths,
                name="Spam",
                nbinsx=30,
                marker_color='#FF6B6B',
                opacity=0.7,
                hoverinfo='name+x+y',
                hoverlabel=dict(bgcolor='#FF6B6B'),
            ))
        
            # Add ham trace
            ham_lengths = df[df['Labels'] == 0]['length']
            fig_dist.add_trace(go.Histogram(
                x=ham_lengths,
                name="Ham",
                nbinsx=30,
                marker_color='#4ECDC4',
                opacity=0.7,
                hoverinfo='name+x+y',
                hoverlabel=dict(bgcolor='#4ECDC4'),
            ))
        
            # Update layout
            fig_dist.update_layout(
                title="Message Length Distribution",
                xaxis_title="Message Length (characters)",
                yaxis_title="Frequency",
                barmode='overlay',
                bargap=0.1,
                hovermode='x unified',
                updatemenus=[
                    dict(
                        type="buttons",
                        showactive=False,
                        buttons=[
                            dict(
                                label="Spam",
                                method="update",
                                args=[{"visible": [True, False]}]
                            ),
                            dict(
                                label="Ham",
                                method="update",
                                args=[{"visible": [False, True]}]
                            ),
                            dict(
                                label="Both",
                                method="update",
                                args=[{"visible": [True, True]}]
                            ),
                        ],
                    )
                ]
            )
        
            st.plotly_chart(fig_dist, use_container_width=True)
        

        with viz_tab2:
            spam_text, ham_text = import_text_corpora()
        
            try:
                col_spam, col_ham = st.columns(2)
            
                # Spam word cloud
                with col_spam:
                    st.subheader("Spam Words")
                    plt.close('all')  # Clear all figures
                
                    wordcloud = WordCloud(
                        width=800, 
                        height=400, 
                        background_color='white',
                        min_font_size=10,
                        max_font_size=50
                    ).generate(spam_text)
                
                    fig_cloud = plt.figure(figsize=(10,5))
                    plt.imshow(wordcloud.to_array())  # Convert to array
                    plt.axis('off')
                    st.pyplot(fig_cloud)
                    plt.close(fig_cloud)
            
                # Ham word cloud
                with col_ham:
                    st.subheader("Non-Spam Words")
                    plt.close('all')  # Clear all figures
                
                    wordcloud = WordCloud(
                        width=800, 
                        height=400, 
                        background_color='white',
                        min_font_size=10,
                        max_font_size=50
                    ).generate(ham_text)
                
                    fig_cloud = plt.figure(figsize=(10,5))
                    plt.imshow(wordcloud.to_array())  # Convert to array
                    plt.axis('off')
                    st.pyplot(fig_cloud)
                    plt.close(fig_cloud)
                
            except Exception as e:
                st.error(f"Error generating word clouds: {str(e)}")   

        with viz_tab3:
            # Confusion Matrix
            conf_matrix = np.array([[12112, 146], [247, 7598]])
            fig_conf = px.imshow(conf_matrix,
                            labels=dict(x="Predicted", y="Actual"),
                            x=['Spam', 'Ham'],
                            y=['Spam', 'Ham'])
            st.plotly_chart(fig_conf, use_container_width=True)

        # Top Spam Triggers
        st.subheader("🚩 Top Spam Triggers")
        triggers_data = pd.DataFrame({
            'Word': ['get', 'r', 'http', 'offer', 'call','free','dear','please','click','customer','day','app','account','valid','today','download','card','apply','visit','xx','till','flat','use','tc','link','sm','code','update','new','order'],
            'Frequency': [8052, 7661, 5696, 5491, 5121, 4667, 3791, 3369, 3305, 2946, 2935, 2865, 2707, 2472, 2260, 2245, 2159, 2112, 2019, 1999, 1998, 1902, 1870, 1870, 1869, 1860, 1769, 1658, 1625, 1622]
        })
        st.bar_chart(triggers_data.set_index('Word'))

    # ---------------------- Tab 3: About ----------------------
    with tab3:
        st.markdown('<div class="tabs-header">ℹ️ About This Project</div>', unsafe_allow_html=True)
        st.write(
            """
            - This **Spam Classifier** leverages a Bernoulli Naive Bayes Model trained on vectorized text data.
            - **Purpose:** To identify spam messages with high accuracy and provide confidence levels for predictions.
            - **Tech Stack:** Python, Streamlit, Scikit-learn, Plotly.
            """
        )
        st.image("https://s38924.pcdn.co/wp-content/uploads/2019/12/6-Tips-to-Reduce-Spam-Form-Entries.jpg", use_column_width=True)
        st.markdown("**Note:** This model is a demo and may not cover all edge cases. Feedback is welcome!")


    # --------------------------- Footer ---------------------------
    st.markdown("""
    ---
    👨‍💻 **Developed as:** Final year Project-I
    """)
if __name__ == "__main__":
    run_app()