from fastapi import FastAPI, File, UploadFile, HTTPException
import tensorflow as tf
import numpy as np
import librosa
import io
import soundfile as sf
from typing import Any
from collections import Counter
from pydub import AudioSegment  # Importing pydub for .aac, .acc, .m4a to .wav conversion

app = FastAPI()

# Load the TFLite model and allocate tensors
try:
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
except Exception as e:
    raise RuntimeError("Failed to load or allocate the TensorFlow Lite model.") from e

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Ensure input shape compatibility
INPUT_SHAPE = input_details[0]['shape']  # (1, X) shape expected for your model
MODEL_INPUT_LENGTH = INPUT_SHAPE[1]  # Length expected by the model (10,000 in this case)

# Label mapping for the output classes
LABELS = ["BPFI", "BPFO", "Normal"]

def preprocess_segment(segment: np.ndarray) -> np.ndarray:
    """
    Preprocess a single audio segment for model inference.
    """
    # Normalize to the range [-1, 1]
    segment = segment / np.max(np.abs(segment))

    # Quantize data to INT8 range [-128, 127] (required for INT8 quantized models)
    segment = np.round(segment * 127).astype(np.int8)
    
    return np.expand_dims(segment, axis=0)  # Match input tensor shape

def extract_features(segment: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract Mel-filterbank energy features with the provided parameters.
    """
    # Parameters
    frame_length = int(0.04 * sr)  # 0.04 seconds in samples
    frame_stride = int(0.04 * sr)  # 0.04 seconds in samples
    n_filters = 200  # Number of Mel filters
    fft_length = 4096  # FFT length
    low_freq = 500  # Low frequency bound (500 Hz)
    high_freq = sr // 2  # High frequency bound (Nyquist frequency)

    # Compute the Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=segment,
                                              sr=sr,
                                              n_fft=fft_length,
                                              hop_length=frame_stride,
                                              win_length=frame_length,
                                              n_mels=n_filters,
                                              fmin=low_freq,
                                              fmax=high_freq)

    # Convert to log scale (Mel-filterbank energy features)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # Flatten the log-mel spectrogram
    mel_features = log_mel_spec.flatten()

    # If the flattened features are less than the model's expected input size, pad them with zeros
    if len(mel_features) < MODEL_INPUT_LENGTH:
        mel_features = np.pad(mel_features, (0, MODEL_INPUT_LENGTH - len(mel_features)), mode='constant')

    # If the flattened features are greater than the expected input size, truncate them
    elif len(mel_features) > MODEL_INPUT_LENGTH:
        mel_features = mel_features[:MODEL_INPUT_LENGTH]

    return mel_features

def convert_to_wav(audio_file: io.BytesIO, file_type: str) -> io.BytesIO:
    """
    Convert various audio formats to .wav using pydub.
    """
    # Load the audio file into AudioSegment
    audio = AudioSegment.from_file(audio_file, format=file_type)
    
    # Create a BytesIO object to hold the WAV data
    wav_io = io.BytesIO()
    
    # Export the audio as a WAV file
    audio.export(wav_io, format="wav")
    wav_io.seek(0)  # Rewind to the beginning of the BytesIO buffer
    
    return wav_io

async def process_audio(file: Any, file_type: str = "wav"):
    """
    Process the audio file sequentially in 2-second segments.
    """
    try:
        if file_type in ["aac", "acc", "m4a"]:  
            # Convert .aac, .acc, .m4a to .wav before processing
            file = convert_to_wav(file, file_type)

        # Load the audio (either original .wav or converted .aac/.acc/.m4a to .wav)
        y, sr = librosa.load(file, sr=None, mono=True)
    except Exception as e:
        raise ValueError("Invalid audio file.") from e

    results = []
    segment_length = sr * 2  # 2 seconds per segment
    num_segments = len(y) // segment_length  # Divide audio by 2-second chunks

    if len(y) % segment_length != 0:
        num_segments += 1

    for i in range(num_segments):
        start = i * segment_length  # 2 seconds per segment
        end = start + segment_length
        segment = y[start:end]

        if len(segment) < segment_length:
            segment = np.pad(segment, (0, segment_length - len(segment)), mode='constant')

        processed_segment = extract_features(segment, sr)
        processed_segment = preprocess_segment(processed_segment)

        interpreter.set_tensor(input_details[0]['index'], processed_segment)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        confidence_scores = prediction[0]  # Array of 3 confidence scores
        a, b, c = confidence_scores

        label = "Unknown"
        if c == -127 and a == 0 and b == 0:
            label = "BFI"
        elif c == -127 and a > 0 and b < 0:
            label = "BFO"
        elif c == -128 and a < 0 and b > 0:
            label = "Normal"
        
        if label == "Unknown":
            label = LABELS[np.argmax(confidence_scores)]

        confidence = float(confidence_scores[np.argmax(confidence_scores)])
        start_time = i * 2
        end_time = (i + 1) * 2

        result = {
            "segment": i + 1,
            "label": label,
            "confidence": confidence,
            "confidence_scores": confidence_scores.tolist(),
            "start_time": start_time,
            "end_time": end_time
        }
        results.append(result)

    return results

@app.post("/predict/")  # Define the API endpoint for prediction
async def predict(file: UploadFile = File(...)):
    try:
        # Determine the file type based on the file extension
        filename = file.filename.lower()
        if filename.endswith(".aac"):
            file_type = "aac"
        elif filename.endswith(".acc"):
            file_type = "acc"
        elif filename.endswith(".m4a"):
            file_type = "m4a"
        elif filename.endswith(".wav"):
            file_type = "wav"
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        # Read and process the audio file
        audio_bytes = await file.read()
        response_generator = process_audio(io.BytesIO(audio_bytes), file_type)

        # Collect all segment results
        results = await response_generator

        # Count the frequency of each label
        labels = [result['label'] for result in results]
        label_counts = Counter(labels)

        # Find the label with the highest count
        most_common_label, count = label_counts.most_common(1)[0]

        # Return the predictions with labels and confidence scores
        return {
            "predictions": results,
            "most_common_label": most_common_label,
            "count": count
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":  # Run the FastAPI app using Uvicorn
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
