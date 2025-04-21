import tensorflow as tf
import numpy as np
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

def process_audio_chunks(audio_path, target_sr=16000, chunk_length=480000):
    # Read audio file
    audio_data = tf.io.read_file(audio_path)
    waveform, sample_rate = tf.audio.decode_wav(audio_data)
    
    # Convert to mono if stereo
    if len(waveform.shape) > 1:
        waveform = tf.reduce_mean(waveform, axis=1)
    
    # Reshape and normalize
    waveform = tf.reshape(waveform, [-1])
    waveform = tf.cast(waveform, tf.float32)
    
    # Resample if needed
    if sample_rate != target_sr:
        waveform = tf.reshape(waveform, [1, -1])
        waveform = tf.image.resize(waveform, [1, int(len(waveform[0]) * target_sr / sample_rate)])[0]
    
    # Convert to numpy
    waveform = waveform.numpy()
    
    # Split into chunks
    chunks = []
    for i in range(0, len(waveform), chunk_length):
        chunk = waveform[i:i + chunk_length]
        # Pad last chunk if needed
        if len(chunk) < chunk_length:
            chunk = np.pad(chunk, (0, chunk_length - len(chunk)))
        chunks.append(chunk)
    
    return chunks, target_sr

def main():
    
    processor = AutoProcessor.from_pretrained("openai/whisper-base", use_auth_token=token)
    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-base", use_auth_token=token)

    # Process audio
    audio_file = "C:/Users/madda/Desktop/coding/Hackattack - 25/resampled_audio.wav"
    chunks, sample_rate = process_audio_chunks(audio_file)
    
    # Process each chunk and combine transcriptions
    full_transcription = []
    
    try:
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            
            # Convert the chunk to features
            input_features = processor(
                chunk,
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).input_features

            # Generate transcription for chunk
            generated_ids = model.generate(input_features)
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            full_transcription.append(transcription.strip())

        # Combine all transcriptions
        final_transcription = " ".join(full_transcription)
        print("\nFull Transcription:")
        print(final_transcription)
        
        # Optionally save to file
        with open("transcription.txt", "w", encoding="utf-8") as f:
            f.write(final_transcription)
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    main()
