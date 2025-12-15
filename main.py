from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio import transforms
import io
import soundfile as sf

labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

num_classes = len(labels)

class CheckAudio(nn.Module):
  def __init__(self):
    super().__init__()
    self.first = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d((4, 4))
    )
    self.second = nn.Sequential(
        nn.Flatten(),
        nn.Linear(64 * 4 * 4, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes)
    )
  def forward(self, x):
    x = x.unsqueeze(1)
    x = self.first(x)
    x = self.second(x)
    return x
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


index_to_label ={ind:lab for ind, lab in enumerate(labels)}
model = CheckAudio()
model.load_state_dict(torch.load('model.pth', map_location=device))
model.to(device)
model.eval()

transform = transforms.MelSpectrogram(
    sample_rate=22050,
    n_mels=128,
    n_fft=2048,
    hop_length=512
)

max_len = 660

def change_audio(waveform, sample_rate):
    if sample_rate != 22050:
        new_sr = transforms.Resample(orig_freq=sample_rate, new_freq=22050)
        waveform = new_sr(torch.tensor(waveform))
    spec = transform(waveform).squeeze(0)
    if spec.shape[1] > max_len:
        spec = spec[:, :max_len]
    if spec.shape[1] < max_len:
        count_len = max_len - spec.shape[1]
        spec = F.pad(spec, (0, count_len))
    return spec

check_music = FastAPI()

@check_music.post('/predict/')
async def predict_audio(file: UploadFile = File(..., )):
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail='Файл пустой')
        wf, sr = sf.read(io.BytesIO(data), dtype='float32')
        wf = torch.tensor(wf).T

        spec = change_audio(wf, sr).unsqueeze(0).to(device)
        with torch.no_grad():
            y_pred = model(spec)
            pred_ind = torch.argmax(y_pred, dim=1).item()
            pred_class = index_to_label[pred_ind]
            return {f'Класс : {pred_class}'}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run(check_music, host='127.0.0.1', port=8001)