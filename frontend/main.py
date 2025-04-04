import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import sys

# Add the project root to the path to import from other modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set page configuration
st.set_page_config(
    page_title="Multi-Task NLP Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download necessary NLTK resources
@st.cache(allow_output_mutation=True)
def download_nltk_resources():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
    except:
        st.warning("Could not download NLTK resources. Some functionality may be limited.")

download_nltk_resources()

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Text preprocessing function
def clean_text(text):
    """
    Clean and preprocess text data:
    - Convert to lowercase
    - Remove URLs, mentions, special characters
    - Remove punctuation
    - Remove stopwords
    - Lemmatize words
    """
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove mentions (@username)
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags
        text = re.sub(r'#\w+', '', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize text
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        
        # Join tokens back into string
        return ' '.join(tokens)
    return ''

# Load model and tokenizers
@st.cache(allow_output_mutation=True)
def load_model_and_tokenizers():
    try:
        # Load the model
        model_path = os.path.join('models', 'checkpoints', 'multi_task_model_20250404_04_1.95.h5')
        model = tf.keras.models.load_model(model_path)
        
        # Load tokenizers
        tokenizer_paths = {
            'emotion': os.path.join('models', 'tokenizers', 'emotion_tokenizer.pickle'),
            'violence': os.path.join('models', 'tokenizers', 'violence_tokenizer.pickle'),
            'hate': os.path.join('models', 'tokenizers', 'hate_tokenizer.pickle')
        }
        
        tokenizers = {}
        for key, path in tokenizer_paths.items():
            with open(path, 'rb') as f:
                tokenizers[key] = pickle.load(f)
                
        return model, tokenizers
    except Exception as e:
        st.error(f"Error loading model or tokenizers: {e}")
        return None, None

# Prediction function
def predict_text(text, model, tokenizers, max_length=50):
    """Make predictions for a given text across all three tasks."""
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Use the emotion tokenizer (you could use any of the three)
    tokenizer = tokenizers['emotion']
    
    # Tokenize and pad the text
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length)
    
    # Make predictions
    emotion_pred, violence_pred, hate_pred = model.predict({
        'emotion_input': padded_sequence,
        'violence_input': padded_sequence,
        'hate_input': padded_sequence
    })
    # Get the class labels
    emotion_classes = ['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise']
    violence_classes = ['Harmful Traditional Practice', 'Physical Violence', 
                       'Economic Violence', 'Emotional Violence', 'Sexual Violence']
    hate_classes = ['Hate Speech', 'Offensive Speech', 'Neither']
    
    # Get the predicted class indices
    emotion_class_idx = np.argmax(emotion_pred[0])
    violence_class_idx = np.argmax(violence_pred[0])
    hate_class_idx = np.argmax(hate_pred[0])
    
    # Create a dictionary of results
    results = {
        'text': text,
        'cleaned_text': cleaned_text,
        'emotion': {
            'label': emotion_classes[emotion_class_idx],
            'confidence': float(emotion_pred[0][emotion_class_idx]),
            'all_probabilities': {emotion_classes[i]: float(emotion_pred[0][i]) 
                               for i in range(len(emotion_classes))}
        },
        'violence': {
            'label': violence_classes[violence_class_idx],
            'confidence': float(violence_pred[0][violence_class_idx]),
            'all_probabilities': {violence_classes[i]: float(violence_pred[0][i]) 
                               for i in range(len(violence_classes))}
        },
        'hate': {
            'label': hate_classes[hate_class_idx],
            'confidence': float(hate_pred[0][hate_class_idx]),
            'all_probabilities': {hate_classes[i]: float(hate_pred[0][i]) 
                              for i in range(len(hate_classes))}
        }
    }
    
    return results

# Main app UI
def main():
    st.title("🤖 Multi-Task NLP Analyzer")
    st.markdown("""
    This application uses a multi-task LSTM neural network to analyze text for:
    * 😊 **Emotions** (Sadness, Joy, Love, Anger, Fear, Surprise)
    * 🚫 **Violence Classification** (Harmful Traditional, Physical, Economic, Emotional, Sexual)
    * 💬 **Hate Speech Detection** (Hate Speech, Offensive Speech, Neither)
    """)
    
    # Load model and tokenizers
    model, tokenizers = load_model_and_tokenizers()
    
    if model is None or tokenizers is None:
        st.error("Failed to load model or tokenizers. Please check the paths and try again.")
        st.stop()
    
    # Create tabs
    st.write("## Text Analysis")
    
    # Text input
    user_input = st.text_area("Type or paste text here:", height=150)
    
    col1, col2 = st.columns([1, 5])
    with col1:
        analyze_button = st.button("Analyze")
    
    # Show sample text options
    with col2:
        sample_texts = {
            "": "",
            "Positive emotion": "I am so happy today! Everything is going well and I feel blessed.",
            "Negative emotion": "I'm feeling really sad and depressed about what happened yesterday.",
            "Violent content": "He threatened to hurt her if she ever tried to leave the relationship.",
            "Hate speech": "I can't stand those people, they should all be banned from our country!",
            "Neutral text": "The weather forecast shows it will be sunny tomorrow with a high of 75 degrees."
        }
        selected_sample = st.selectbox("Or try a sample:", options=list(sample_texts.keys()))
        if selected_sample and selected_sample != "":
            user_input = sample_texts[selected_sample]
    
    # Process and display results
    if analyze_button and user_input:
        with st.spinner("Analyzing..."):
            # Make predictions
            results = predict_text(user_input, model, tokenizers)
            
            # Display results in colored boxes
            st.markdown("### Analysis Results")
            
            # Display the cleaned text
            st.markdown("**Preprocessed text:**")
            st.info(results['cleaned_text'])
            
            # Create columns for results
            col1, col2, col3 = st.columns(3)
            
            # Emotion results
            with col1:
                st.markdown("#### Emotion Detection")
                emotion_label = results['emotion']['label']
                emotion_conf = results['emotion']['confidence']
                
                # Choose color based on emotion
                color = {
                    'Joy': 'rgba(255, 215, 0, 0.2)',    # Gold
                    'Love': 'rgba(255, 105, 180, 0.2)', # Pink
                    'Anger': 'rgba(255, 0, 0, 0.2)',    # Red
                    'Sadness': 'rgba(0, 0, 255, 0.2)',  # Blue
                    'Fear': 'rgba(128, 0, 128, 0.2)',   # Purple
                    'Surprise': 'rgba(0, 255, 255, 0.2)' # Cyan
                }.get(emotion_label, 'rgba(200, 200, 200, 0.2)')
                
                st.markdown(
                    f"""
                    <div style="padding: 20px; border-radius: 10px; background-color: {color};">
                        <h4 style="margin:0;">{emotion_label}</h4>
                        <p style="font-size: 24px; margin:0; font-weight: bold;">
                            {emotion_conf:.1%}
                        </p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Show all emotion probabilities
                st.markdown("##### All emotions:")
                for emotion, prob in results['emotion']['all_probabilities'].items():
                    st.markdown(f"- {emotion}: {prob:.1%}")
            
            # Violence detection results
            with col2:
                st.markdown("#### Violence Detection")
                violence_label = results['violence']['label']
                violence_conf = results['violence']['confidence']
                
                # Color intensity based on severity
                severity_level = {
                    'Harmful Traditional Practice': 0.6,
                    'Physical Violence': 0.9,
                    'Economic Violence': 0.5,
                    'Emotional Violence': 0.7,
                    'Sexual Violence': 1.0
                }.get(violence_label, 0.3)
                
                color = f"rgba(255, 0, 0, {severity_level * 0.2})"
                
                st.markdown(
                    f"""
                    <div style="padding: 20px; border-radius: 10px; background-color: {color};">
                        <h4 style="margin:0;">{violence_label}</h4>
                        <p style="font-size: 24px; margin:0; font-weight: bold;">
                            {violence_conf:.1%}
                        </p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Show all violence probabilities
                st.markdown("##### All violence types:")
                for violence, prob in results['violence']['all_probabilities'].items():
                    st.markdown(f"- {violence}: {prob:.1%}")
            
            # Hate speech results
            with col3:
                st.markdown("#### Hate Speech Detection")
                hate_label = results['hate']['label']
                hate_conf = results['hate']['confidence']
                
                # Color based on category
                color = {
                    'Hate Speech': 'rgba(255, 0, 0, 0.2)',
                    'Offensive Speech': 'rgba(255, 165, 0, 0.2)',
                    'Neither': 'rgba(0, 128, 0, 0.2)'
                }.get(hate_label, 'rgba(200, 200, 200, 0.2)')
                
                st.markdown(
                    f"""
                    <div style="padding: 20px; border-radius: 10px; background-color: {color};">
                        <h4 style="margin:0;">{hate_label}</h4>
                        <p style="font-size: 24px; margin:0; font-weight: bold;">
                            {hate_conf:.1%}
                        </p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Show all hate speech probabilities
                st.markdown("##### All categories:")
                for category, prob in results['hate']['all_probabilities'].items():
                    st.markdown(f"- {category}: {prob:.1%}")
    
    st.write("---")
    
    # About the model section
    st.write("## About the Model")
    st.markdown("""
    ### Multi-Task LSTM Model for Text Analysis
    
    This application uses a deep learning model that performs three different NLP tasks simultaneously:
    
    1. **Emotion Detection**: Classifies text into one of six emotional categories
       - Sadness, Joy, Love, Anger, Fear, Surprise
    
    2. **Violence Classification**: Identifies different types of violence mentioned in text
       - Harmful Traditional Practice, Physical Violence, Economic Violence, Emotional Violence, Sexual Violence
    
    3. **Hate Speech Detection**: Determines if text contains hate speech, offensive language, or neither
       - Hate Speech, Offensive Speech, Neither
    
    ### Model Architecture
    
    The model uses a multi-task learning approach with:
    - Shared text embedding layer
    - Bidirectional LSTM layers for text feature extraction
    - Task-specific dense output layers for each classification task
    
    ### Training Data
    
    The model was trained on three different datasets:
    - Emotion Data: https://www.kaggle.com/datasets/nelgiriyewithana/emotions
    - Violence Data: https://www.kaggle.com/datasets/gauravduttakiit/gender-based-violence-tweet-classification
    - Hate Speech Data: https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset
    """)
    
    st.write("---")
    
    # Sample texts section
    st.write("## Sample Texts to Try")
    
    samples = [
        {
            "title": "Positive Emotion",
            "text": "I'm feeling so happy today! Everything is going well in my life and I'm grateful for all the blessings.",
            "description": "This text should be classified with a positive emotion like Joy or Love."
        },
        {
            "title": "Negative Emotion",
            "text": "I'm terrified of what might happen. The situation keeps getting worse and I don't know what to do.",
            "description": "This text should be classified with a negative emotion like Fear or Sadness."
        },
        {
            "title": "Violent Content",
            "text": "He threatened to hit her if she tried to leave the relationship again. She feels trapped and scared.",
            "description": "This text contains references to physical violence and emotional abuse."
        },
        {
            "title": "Potentially Offensive",
            "text": "That was such a stupid decision! Only an idiot would think that could work. What a waste of time.",
            "description": "This text contains potentially offensive language but isn't necessarily hate speech."
        },
        {
            "title": "Neutral Text",
            "text": "The meeting is scheduled for tomorrow at 3 PM. Please bring your laptop and the quarterly report.",
            "description": "This text is neutral and shouldn't trigger strong classifications in any category."
        }
    ]
    
    for i, sample in enumerate(samples):
        st.subheader(f"{i+1}. {sample['title']}")
        st.text_area(f"Sample {i+1}", sample["text"], height=80, key=f"sample_{i}")
        st.markdown(f"*{sample['description']}*")
        if st.button(f"Analyze Sample {i+1}", key=f"btn_{i}"):
            with st.spinner("Analyzing..."):
                results = predict_text(sample["text"], model, tokenizers)
                
                # Display results
                st.markdown("#### Results:")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**Emotion:** {results['emotion']['label']} ({results['emotion']['confidence']:.1%})")
                with col2:
                    st.markdown(f"**Violence:** {results['violence']['label']} ({results['violence']['confidence']:.1%})")
                with col3:
                    st.markdown(f"**Hate Speech:** {results['hate']['label']} ({results['hate']['confidence']:.1%})")
        
        st.markdown("---")

if __name__ == "__main__":
    main()