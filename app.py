import streamlit as st
import torch
import torchaudio
import torch.nn as nn
from nnAudio.features import CQT
import numpy as np
import io
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mix_down(audio):
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdims=True)
    return audio

def truncate_padding_audio(audio, target_length, sr):
    max_length = target_length * sr
    audio_length = audio.size(1)

    if audio_length > max_length:
        start_frame = (audio_length - max_length) // 2
        return audio[:, start_frame:start_frame + max_length]
    else:
        padding = (0, max_length - audio_length)
        return nn.functional.pad(audio, padding, mode='constant', value=0.0)

def cqt_feature_extraction(audio, sample_rate=16000, n_bins=84, hop_length=512):
    cqt_transform = CQT(sr=sample_rate, n_bins=n_bins, hop_length=hop_length).to(device)
    audio = audio.to(device)
    return cqt_transform(audio)

def create_lightcnn(num_classes=79077, filters=[48, 96, 192, 128], kernel_sizes=[5, 3, 3, 3], 
                    strides=[1, 1, 1, 1], paddings=[2, 1, 1, 1], leaky_relu_alpha=0.2, dropout_prob=0.3):
    class mfm(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
            super(mfm, self).__init__()
            self.out_channels = out_channels
            if type == 1:
                self.filter = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
            else:
                self.filter = nn.Linear(in_channels, 2 * out_channels)

        def forward(self, x):
            x = self.filter(x)
            out = torch.split(x, self.out_channels, 1)
            return torch.max(out[0], out[1])

    class group(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
            super(group, self).__init__()
            self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
            self.conv = mfm(in_channels, out_channels, kernel_size, stride, padding)

        def forward(self, x):
            x = self.conv_a(x)
            x = self.conv(x)
            return x

    class network_9layers(nn.Module):
        def __init__(self):
            super(network_9layers, self).__init__()

            self.conv1 = nn.Sequential(
                mfm(1, filters[0], kernel_sizes[0], strides[0], paddings[0]), 
                nn.BatchNorm2d(filters[0]),
                nn.LeakyReLU(leaky_relu_alpha),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) 
            )

            self.group1 = nn.Sequential(
                group(filters[0], filters[1], kernel_sizes[1], strides[1], paddings[1]),
                nn.BatchNorm2d(filters[1]),
                nn.LeakyReLU(leaky_relu_alpha),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) 
            )

            self.group2 = nn.Sequential(
                group(filters[1], filters[2], kernel_sizes[2], strides[2], paddings[2]), 
                nn.BatchNorm2d(filters[2]),
                nn.LeakyReLU(leaky_relu_alpha),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) 
            )

            self.group3 = nn.Sequential(
                group(filters[2], filters[3], kernel_sizes[3], strides[3], paddings[3]), 
                nn.BatchNorm2d(filters[3]),
                nn.LeakyReLU(leaky_relu_alpha),
            )

            self.group4 = nn.Sequential(
                group(filters[3], filters[3], kernel_sizes[3], strides[3], paddings[3]), 
                nn.BatchNorm2d(filters[3]),
                nn.LeakyReLU(leaky_relu_alpha),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
            )

            self.fc1 = nn.Sequential(
                mfm(7680, 128, type=0), 
                nn.BatchNorm1d(128),
                nn.LeakyReLU(leaky_relu_alpha),
                nn.Dropout(dropout_prob)
            )

            self.fc2 = nn.Linear(128, num_classes)
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.group1(x)
            x = self.group2(x)
            x = self.group3(x)
            x = self.group4(x)
            x = x.view(x.size(0), -1) 
            x = self.fc1(x) 
            out = self.fc2(x)
            return out, x

    return network_9layers()

model_paths = [
    "tanpa_augmentasi.pth",
    "shufflemix.pth",
    "mixup.pth",
    "cutmix.pth"
]

models = []
for model_path in model_paths:
    model = create_lightcnn(num_classes=2, filters=[48, 96, 192, 128], leaky_relu_alpha=0.1, dropout_prob=0.5).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    models.append(model)

# Fungsi untuk memprediksi audio dengan model
def predict_audio(file, model, sample_rate=16000, target_length=5):
    waveform, sr = torchaudio.load(file)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)

    waveform = mix_down(waveform)
    waveform = truncate_padding_audio(waveform, target_length, sample_rate)
    features = cqt_feature_extraction(waveform, sample_rate)
    features = features.unsqueeze(0)

    with torch.no_grad():
        outputs, _ = model(features)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        pred = torch.argmax(outputs, dim=1).item()

    return pred, probs

# Streamlit UI
st.title("Deteksi Deepfake pada Audio")

st.write(
    """
    Unggah atau rekam file audio untuk mendeteksi apakah audio tersebut asli atau deepfake. 
    """
)

# File uploader or recording
uploaded_files = st.file_uploader("Pilih file audio", type=["wav", "mp3", "flac"], accept_multiple_files=True)

if uploaded_files:
    st.write("Files uploaded:")
    results = []
    for audio_file in uploaded_files:
        # Menampilkan audio player untuk setiap file
        st.audio(audio_file)
        st.write(f"### Hasil Prediksi untuk {audio_file.name}")
        
        with st.spinner(f"Memproses {audio_file.name}..."):
            audio_bytes = audio_file.read()
            audio_file.seek(0)
            all_preds = []
            all_probs = []
            for model in models:
                pred, probs = predict_audio(io.BytesIO(audio_bytes), model)
                all_preds.append(pred)
                all_probs.append(probs)

        # Menyusun hasil prediksi untuk tabel
        file_results = []
        for idx, model in enumerate(models):
            if all_preds[idx] == 1:
                prediction = "Deepfake"
            else:
                prediction = "Asli"
            file_results.append({
                "Model": model_paths[idx].split('/')[-1].replace('_', ' ').replace('.pth', ''),
                "Prediksi": prediction,
                "Probabilitas Asli": f"{all_probs[idx][0][0]:.4f}",
                "Probabilitas Deepfake": f"{all_probs[idx][0][1]:.4f}"
            })

        # Membuat DataFrame dari hasil dan memberikan indeks mulai dari 1
        df_results = pd.DataFrame(file_results)
        df_results.index += 1  # Mengubah indeks menjadi mulai dari 1

        # Menampilkan tabel hasil prediksi untuk setiap file
        st.table(df_results)

else:
    st.write("Unggah file audio untuk mendapatkan prediksi.")
