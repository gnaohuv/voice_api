import random
import numpy as np
from fastapi import FastAPI, UploadFile, File
from deep_speaker.audio import read_mfcc
from deep_speaker.batcher import sample_from_mfcc
from deep_speaker.constants import SAMPLE_RATE, NUM_FRAMES
from deep_speaker.conv_models import DeepSpeakerModel
from deep_speaker.test import batch_cosine_similarity

# Reproducible results.
np.random.seed(123)
random.seed(123)

# Load the Deep Speaker model.
model = DeepSpeakerModel()
model.m.load_weights("H:/ResCNN_triplet_training_checkpoint_265.h5", by_name=True)

app = FastAPI()


def extract_embedding(audio_path):
    np.random.seed(123)
    random.seed(123)
    """Hàm dùng để trích xuất embedding từ file âm thanh."""
    mfcc = sample_from_mfcc(read_mfcc(audio_path, SAMPLE_RATE), NUM_FRAMES)
    embedding = model.m.predict(np.expand_dims(mfcc, axis=0))
    return embedding


@app.post("/compare/")
async def compare_audio(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    # Lưu hai file âm thanh tạm thời
    with open("audio1.wav", "wb") as f1, open("audio2.wav", "wb") as f2:
        f1.write(await file1.read())
        f2.write(await file2.read())

    # Trích xuất embedding từ file âm thanh
    embedding1 = extract_embedding("audio1.wav")
    embedding2 = extract_embedding("audio2.wav")

    # Tính chỉ số cosine similarity giữa hai embedding
    similarity = batch_cosine_similarity(embedding1, embedding2)

    return {"similarity": float(similarity)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
