import os, tempfile, json, queue, subprocess
import soundfile as sf
import sounddevice as sd
from scipy.signal import resample
from vosk import Model, KaldiRecognizer
import config

def downsample(wav_path, target_sr=16000):
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        n_samples = int(len(audio) * target_sr / sr)
        audio = resample(audio, n_samples)
    return audio, target_sr

def convert_to_wav(audio_file_path):
    try:
        file_ext = os.path.splitext(audio_file_path)[1].lower()
        if file_ext == '.wav':
            return audio_file_path
        
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_wav.close()
        
        cmd = [
            'ffmpeg', '-i', audio_file_path,
            '-ar', '16000',  # sample rate 16kHz
            '-ac', '1',      # mono channel
            '-sample_fmt', 's16',  # 16bit
            '-y',            # overwrite
            temp_wav.name
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            os.unlink(temp_wav.name)
            return None
            
        return temp_wav.name
    
    except Exception as e:
        print(f"Error converting audio file: {e}")
        if 'temp_wav' in locals() and os.path.exists(temp_wav.name):
            os.unlink(temp_wav.name)
        return None


def recognize_speech_from_file(stt_model, audio_file_path):
    wav_path = convert_to_wav(audio_file_path)
    if wav_path is None:
        return ""
    
    audio, sr = downsample(wav_path, target_sr=16000)
    rec = KaldiRecognizer(stt_model, sr)

    results = []
    chunk_size = 4000
    for i in range(0, len(audio), chunk_size):
        data = (audio[i:i+chunk_size] * 32767).astype('int16').tobytes()
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            results.append(res.get('text', ''))

    final_res = json.loads(rec.FinalResult())
    results.append(final_res.get('text', ''))

    transcript = ' '.join(results).strip()

    if wav_path != audio_file_path:
        os.unlink(wav_path)

    return transcript


def recognize_from_mic(stt_model, duration_sec=5, samplerate=16000, blocksize=8000):

    q = queue.Queue()

    def _callback(indata, frames, time, status):
        if status:
            print(status, flush=True)
        q.put(indata.copy().tobytes())

    rec = KaldiRecognizer(stt_model, samplerate)
    rec.SetWords(True)

    with sd.InputStream(callback=_callback, channels=1, samplerate=samplerate, blocksize=blocksize, dtype='int16'):
        print("Recording...")
        num_blocks = int(samplerate * duration_sec / blocksize)
        for _ in range(num_blocks):
            data = q.get()
            if rec.AcceptWaveform(data):
                pass
        
        results = json.loads(rec.FinalResult())
        print("Recording complete.")

    transcript = results.get('text', '').strip()
    
    return transcript


def main():
    model = Model(config.STT_PATH)
    for i in range(25):
        config.AUDIO_PATH = os.path.join(config.AUDIO_DIR, f"output{i+1}.mp3")

        if config.AUDIO_PATH:
            transcript = recognize_speech_from_file(model, config.AUDIO_PATH)
            print(f"Transcript: {transcript}")
        else:
            audio_files = []
            if os.path.exists(config.AUDIO_DIR):
                for file in os.listdir(config.AUDIO_DIR):
                    if file.endswith(('.wav', '.mp3', '.flac', '.aiff')):
                        audio_files.append(os.path.join(config.AUDIO_DIR, file))
            if audio_files:
                print(f"Using audio file: {audio_files[0]}")
                transcript = recognize_speech_from_file(model, audio_files[0])
                print(f"Transcript: {transcript}")

if __name__ == "__main__":
    main()