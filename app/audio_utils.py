import os
import librosa
import librosa.display
import matplotlib
# 使用非互動式後端，這在伺服器環境下至關重要
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from datetime import datetime
from .ai_model import run_inference
import soundfile as sf

from scipy.signal import butter, sosfiltfilt, decimate, get_window, lfilter, hilbert
from numpy.fft import fft, fftfreq

# --- 常數與輔助函式 ---

CLASSIC_DEMON_PARAMS = {
    'BANDPASS_LOW': 2000, 'BANDPASS_HIGH': 7500, 'DOWNSAMPLE_RATE': 2000,
    'WINDOW_SIZE': 2048, 'WINDOW_OVERLAP_RATIO': 0.95, 'WINDOW_TYPE': 'hann',
    'FREQ_YLIM': 200
}

def _bandpass_filter(signal, fs, lowcut, highcut, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfiltfilt(sos, signal)

def _square_law_demodulate(signal):
    return signal ** 2

# --- 核心繪圖函式 ---

def save_spectrogram(y, sr, out_path_display, out_path_training, spec_type='mel'):
    """根據類型，產生並儲存顯示版和訓練版的頻譜圖。"""
    if spec_type == 'classic_demon':
        save_classic_demon_plot(y, sr, out_path_display, out_path_training)
        return
    elif spec_type == 'envelope_spectrum':
        save_envelope_spectrum_plot(y, sr, out_path_display, out_path_training)
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    
    if spec_type == 'mel':
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr/2)
        S_db = librosa.power_to_db(S, ref=np.max)
        display_data = S_db
    elif spec_type == 'stft':
        # ===== vvvv 優化點 1：控制 STFT 計算量 vvvv =====
        # 透過明確設定 n_fft 和 hop_length，可以大幅減少 STFT 輸出的資料矩陣大小，
        # 這是解決 Matplotlib 處理長音檔時卡頓的關鍵。
        n_fft = 1024      # 使用較小的 FFT 窗口
        hop_length = 512  # 使用較大的步長
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        # ===== ^^^^ 優化點 1 結束 ^^^^ =====
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        display_data = S_db
    else:
        plt.close(fig)
        return

    librosa.display.specshow(display_data, sr=sr, x_axis='time', y_axis='mel' if spec_type=='mel' else 'hz', ax=ax)
    ax.set_title(f'{spec_type.capitalize()} Spectrogram')
    fig.colorbar(ax.collections[0], ax=ax, format='%+2.0f dB')
    plt.tight_layout()
    
    # ===== vvvv 優化點 2：控制圖片輸出解析度 vvvv =====
    # 設定合理的 DPI 可以防止 Matplotlib 產生過大的圖片檔案，節省磁碟空間和記憶體。
    plt.savefig(out_path_display, dpi=100)
    
    fig.clear()
    ax = fig.add_subplot(111)
    librosa.display.specshow(display_data, sr=sr, ax=ax)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(out_path_training, bbox_inches='tight', pad_inches=0, dpi=100)
    # ===== ^^^^ 優化點 2 結束 ^^^^ =====
    plt.close(fig)

def save_classic_demon_plot(segment, sr, out_path_display, out_path_training):
    params = CLASSIC_DEMON_PARAMS
    nyquist = sr / 2
    bandpass_high = min(params['BANDPASS_HIGH'], nyquist * 0.99)
    if params['BANDPASS_LOW'] >= bandpass_high: return
    filtered = _bandpass_filter(segment, sr, params['BANDPASS_LOW'], bandpass_high)
    demodulated = _square_law_demodulate(filtered)
    decimation_factor = max(1, int(sr / params['DOWNSAMPLE_RATE']))
    decimated_signal = decimate(demodulated, decimation_factor)
    processed_signal = decimated_signal - np.mean(decimated_signal)
    fs_demo = sr // decimation_factor
    window = get_window(params['WINDOW_TYPE'], params['WINDOW_SIZE'])
    S, freqs, times, _ = plt.specgram(processed_signal, NFFT=params['WINDOW_SIZE'], Fs=fs_demo, window=window, noverlap=int(params['WINDOW_SIZE'] * params['WINDOW_OVERLAP_RATIO']))
    S_db = 10 * np.log10(S + 1e-9)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pcolormesh(times, freqs, S_db, cmap='viridis', shading='auto')
    ax.set_ylim(0, params['FREQ_YLIM'])
    ax.set_ylabel('Modulation Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_title("Classic DEMON Spectrogram (2D)")
    fig.colorbar(ax.collections[0], ax=ax, label='Amplitude (dB)')
    plt.tight_layout()
    plt.savefig(out_path_display, dpi=100)
    
    fig.clear()
    ax = fig.add_subplot(111)
    ax.pcolormesh(times, freqs, S_db, cmap='viridis', shading='auto')
    ax.set_ylim(0, params['FREQ_YLIM'])
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(out_path_training, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)

def save_envelope_spectrum_plot(segment, sr, out_path_display, out_path_training):
    try:
        nyquist = sr / 2
        bp_low, bp_high = 2000, min(20000, nyquist * 0.99)
        if bp_low >= bp_high: return
        
        b, a = butter(4, [bp_low, bp_high], btype='band', fs=sr)
        segment_filt = lfilter(b, a, segment)
        envelope = np.abs(hilbert(segment_filt))
        
        N = len(envelope)
        if N == 0: return
        
        yf = fft(envelope - np.mean(envelope))
        xf = fftfreq(N, 1 / sr)
        half_N = N // 2
        xf_positive, yf_positive = xf[:half_N], 2.0/N * np.abs(yf[:half_N])
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xf_positive, yf_positive)
        ax.set_title('Envelope Spectrum (DEMON 1D)')
        ax.set_xlabel('Modulation Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.grid(True)
        ax.set_xlim(0, 300)
        plt.tight_layout()
        plt.savefig(out_path_display, dpi=100)

        ax.clear()
        ax.plot(xf_positive, yf_positive)
        ax.set_xlim(0, 300)
        ax.axis('off')
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.savefig(out_path_training, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close(fig)
    except Exception as e:
        print(f"處理包絡線頻譜時發生錯誤: {e}")

# --- 記憶體優化處理流程 ---

def process_large_audio(filepath, result_dir, spec_type, segment_duration=2.0, overlap_ratio=0.5, target_sr=None, is_mono=True, progress_callback=None):
    """
    以逐段精確載入的方式處理大型音訊檔案，確保所有片段長度一致且節省記憶體。
    """
    all_results = []
    basename = f"{os.path.splitext(os.path.basename(filepath))[0]}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    try:
        info = sf.info(filepath)
        original_sr = info.samplerate
        sr = target_sr if target_sr else original_sr
        total_samples = info.frames

        frame_length = int(segment_duration * sr)
        hop_length = int(frame_length * (1 - overlap_ratio))

        if total_samples < frame_length:
            print("警告：音訊檔案總長度小於設定的單一片段長度。")
            y_segment, _ = librosa.load(filepath, sr=sr, mono=is_mono)
            if len(y_segment) < frame_length:
                y_segment = np.pad(y_segment, (0, frame_length - len(y_segment)))
            segments_to_process = [(0, y_segment)]
        else:
            start_samples = np.arange(0, total_samples - frame_length + 1, hop_length)
            last_segment_start = total_samples - frame_length
            if last_segment_start > start_samples[-1]:
                 start_samples = np.append(start_samples, last_segment_start)
            segments_to_process = [(start_s, None) for start_s in np.unique(start_samples)]

        total_segments = len(segments_to_process)
        for i, (start_s, preloaded_segment) in enumerate(segments_to_process):
            if preloaded_segment is not None:
                y_segment = preloaded_segment
            else:
                y_segment, _ = librosa.load(
                    filepath, 
                    sr=sr, 
                    mono=is_mono, 
                    offset=start_s / sr, 
                    duration=segment_duration
                )

            if len(y_segment) < frame_length:
                y_segment = np.pad(y_segment, (0, frame_length - len(y_segment)))
            
            audio_filename = f"{basename}_part{i}.wav"
            display_spec_filename = f"{basename}_spec_display_{i}.png"
            training_spec_filename = f"{basename}_spec_training_{i}.png"

            audio_path = os.path.join(result_dir, audio_filename)
            display_spec_path = os.path.join(result_dir, display_spec_filename)
            training_spec_path = os.path.join(result_dir, training_spec_filename)
            
            wavfile.write(audio_path, sr, (y_segment.T if y_segment.ndim > 1 else y_segment * 32767).astype(np.int16))
            
            mono_segment = librosa.to_mono(y_segment) if y_segment.ndim > 1 else y_segment
            save_spectrogram(mono_segment, sr, display_spec_path, training_spec_path, spec_type)
            
            all_results.append({
                'audio': audio_filename,
                'display_spectrogram': display_spec_filename,
                'training_spectrogram': training_spec_filename,
                'detections': run_inference(training_spec_path)
            })
            
            if progress_callback:
                progress_callback(i + 1, total_segments)
        
    except Exception as e:
        print(f"處理大型音訊檔案時發生錯誤: {e}")
        raise e
        
    return all_results
