from moviepy.editor import VideoFileClip
import librosa
import soundfile as sf
import os

# Step 1: Extract audio from video
video_path = "E:/Downloads/Machine Learning Tutorial Python - 5_ Save Model Using Joblib And Pickle.mp4"
audio_path = "extracted_audio.wav"

video = VideoFileClip(video_path)
video.audio.write_audiofile(audio_path)

audio, original_sr = librosa.load(audio_path, sr=None)

# Resample to 16,000 Hz
resampled_audio = librosa.resample(audio, orig_sr=original_sr, target_sr=16000)

# Save resampled audio
sf.write("resampled_audio.wav", resampled_audio, 16000)
print("Audio successfully resampled to 16,000 Hz!")
os.remove(audio_path)  # Remove the original audio file


