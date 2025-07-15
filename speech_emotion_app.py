import os
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import streamlit as st
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns

# ----- Your dataset path -----
DATA_PATH = r"C:\Users\USER\OneDrive\Desktop\CodeAlpha\Data"

# Emotion mapping
emotion_map = {
    'angry': 'Angry',
    'disgust': 'Disgust',
    'fear': 'Fear',
    'happy': 'Happy',
    'ps': 'Pleasant Surprise',
    'sad': 'Sad',
    'neutral': 'Neutral'
}

# Descriptions for UI info
emotion_descriptions = {
    'Angry': 'Feeling or showing strong annoyance, displeasure, or hostility.',
    'Disgust': 'A feeling of revulsion or profound disapproval.',
    'Fear': 'An unpleasant emotion caused by the threat of danger or harm.',
    'Happy': 'Feeling or showing pleasure or contentment.',
    'Pleasant Surprise': 'A feeling of unexpected happiness or delight.',
    'Sad': 'Feeling sorrow or unhappiness.',
    'Neutral': 'Neither positive nor negative emotion; calm or indifferent.'
}

def get_emotion(filename):
    for key in emotion_map:
        if key in filename.lower():
            return emotion_map[key]
    return None

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print(f"Error: {file_path} - {e}")
        return None

# Dataset class
class EmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Model definition
class EmotionModel(nn.Module):
    def __init__(self, num_classes):
        super(EmotionModel, self).__init__()
        self.fc1 = nn.Linear(40, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        return self.out(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_data_and_train(epochs):
    features, labels = [], []

    for root, _, files in os.walk(DATA_PATH):
        for filename in files:
            if filename.endswith(".wav"):
                emotion = get_emotion(filename)
                if emotion:
                    path = os.path.join(root, filename)
                    feat = extract_features(path)
                    if feat is not None:
                        features.append(feat)
                        labels.append(emotion)

    X = np.array(features, dtype=np.float32)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    # Show sample counts per emotion in sidebar
    counts = {label: list(labels).count(label) for label in label_encoder.classes_}
    st.sidebar.subheader("Dataset Sample Counts")
    st.sidebar.bar_chart(counts)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    train_dataset = EmotionDataset(X_train, y_train)
    test_dataset = EmotionDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = EmotionModel(num_classes=len(label_encoder.classes_)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    progress_bar = st.progress(0)
    loss_text = st.empty()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        loss_text.text(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        progress_bar.progress((epoch + 1) / epochs)

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_true.extend(y_batch.numpy())
            y_pred.extend(preds)

    acc = accuracy_score(y_true, y_pred)
    st.sidebar.write(f"✅ Test Accuracy: {acc * 100:.2f}%")

    # Show confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap='Blues', ax=ax)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.sidebar.pyplot(fig)

    return model, label_encoder

def predict_emotion_from_audio(model, label_encoder, audio_path):
    model.eval()
    feat = extract_features(audio_path)
    if feat is not None:
        feat_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(feat_tensor)
            predicted = torch.argmax(output, dim=1).item()
            return label_encoder.inverse_transform([predicted])[0]
    return "Unknown"

# -------- STREAMLIT APP --------

st.title("Speech Emotion Recognition 🎤➡️😊")

st.sidebar.title("About")
st.sidebar.info(
    """
    This app uses a PyTorch neural network to recognize emotions from speech audio.
    Upload a .wav file to see the predicted emotion.
    """
)

st.sidebar.title("Emotion Labels & Descriptions")
for emo, desc in emotion_descriptions.items():
    st.sidebar.markdown(f"**{emo}**: {desc}")

epochs = st.sidebar.slider("Training Epochs", min_value=10, max_value=100, value=50, step=5)

if 'model' not in st.session_state or st.sidebar.button("Retrain Model"):
    with st.spinner('Training model, please wait...'):
        model, label_encoder = load_data_and_train(epochs)
        st.session_state['model'] = model
        st.session_state['label_encoder'] = label_encoder
else:
    model = st.session_state['model']
    label_encoder = st.session_state['label_encoder']

uploaded_file = st.file_uploader("Upload a speech audio (.wav) file", type=["wav"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        temp_file_path = tmp_file.name

    st.audio(temp_file_path, format='audio/wav')

    prediction = predict_emotion_from_audio(model, label_encoder, temp_file_path)
    st.markdown(f"### Predicted Emotion: **{prediction}**")