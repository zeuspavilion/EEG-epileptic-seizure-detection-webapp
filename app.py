import os
import json
import numpy as np
import mne
import tensorflow as tf
from flask import Flask, request, jsonify
from scipy.signal import butter, filtfilt, iirnotch, resample_poly
import tempfile
from threading import Lock

# Suppress TF logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- 1. Configuration ---
# "Harvard Standard": The model is trained on 256Hz. We must match this strictly.
SAMPLING_RATE = 256  
LOWCUT = 0.5
HIGHCUT = 40.0
WINDOW_SECONDS = 2
OVERLAP_PERCENT = 0.5
STEP_SECONDS = WINDOW_SECONDS * (1 - OVERLAP_PERCENT)
N_CHANNELS = 18

# --- Safe constants ---
VOLTS_TO_UV = 1e6
CLIP_LIMIT_UV = 1000.0  
EPS = 1e-8

# --- Model caching ---
_model = None
_model_lock = Lock()

def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                try:
                    # Ensure this file exists in your directory
                    _model = tf.keras.models.load_model('BEST_MODEL_cnn_lstm.h5')
                    print("* Model loaded successfully!")
                except Exception as e:
                    print(f"!!! CRITICAL: Error loading model: {e}")
                    return None
    return _model

# --- 2. Advanced Signal Processing (The "Harvard" Upgrade) ---

def mad(x):
    x = np.asarray(x)
    med = np.median(x)
    return np.median(np.abs(x - med)) + 1e-12

def smooth_probabilities(prob, ma_window=5, median_k=3):
    if len(prob) == 0: return prob
    kernel = np.ones(ma_window) / ma_window
    smoothed = np.convolve(prob, kernel, mode='same')
    from scipy.signal import medfilt
    if median_k % 2 == 0: median_k += 1
    filtered = medfilt(smoothed, kernel_size=median_k)
    return filtered

def apply_advanced_filtering(data, fs):
    """
    Invigilator's Standard Filter Chain:
    1. Notch Filter 50Hz (India/Europe Power)
    2. Notch Filter 60Hz (US Power - just in case)
    3. Bandpass 0.5Hz - 40Hz (The Guillotine for artifacts)
    """
    nyq = 0.5 * fs
    
    # A. The Twin-Notch (Surgical Removal of Mains Hum)
    # 50 Hz Notch
    if fs > 100: # Can only notch if fs is high enough
        b_notch, a_notch = iirnotch(50.0, 30.0, fs)
        data = filtfilt(b_notch, a_notch, data, axis=-1)
        
        # 60 Hz Notch
        b_notch60, a_notch60 = iirnotch(60.0, 30.0, fs)
        data = filtfilt(b_notch60, a_notch60, data, axis=-1)

    # B. The Bandpass (Butterworth Order 5)
    low = LOWCUT / nyq
    high = HIGHCUT / nyq
    
    # Safety check for Nyquist
    if high >= 1.0: high = 0.99 

    b, a = butter(5, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)

def segment_data_continuous(data, window_seconds, overlap_percent):
    window_samples = int(window_seconds * SAMPLING_RATE)
    step_samples = int(window_samples * (1 - overlap_percent))
    num_channels = data.shape[0]
    num_samples = data.shape[1]

    if num_samples < window_samples:
        return np.empty((0, num_channels, window_samples))

    num_windows = (num_samples - window_samples) // step_samples + 1
    windows = np.zeros((num_windows, num_channels, window_samples))

    for i in range(num_windows):
        start_index = i * step_samples
        end_index = start_index + window_samples
        windows[i] = data[:, start_index:end_index]
    return windows

def normalize_windows(windows):
    # Standard Score (Z-score) normalization
    X_mean = windows.mean(axis=-1, keepdims=True)
    X_std = windows.std(axis=-1, keepdims=True)
    return (windows - X_mean) / (X_std + EPS)

def save_uploaded_file_safe(file_storage_obj, suffix=".edf"):
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_name = tmp.name
    try:
        file_storage_obj.save(tmp_name)
        tmp.close()
        return tmp_name
    except Exception:
        try:
            tmp.close()
            os.remove(tmp_name)
        except Exception:
            pass
        raise

def safe_clip_and_unit_convert(eeg_data, data_unit='V'):
    if data_unit == 'V':
        eeg_uv = eeg_data * VOLTS_TO_UV
    elif data_unit == 'uV':
        eeg_uv = eeg_data
    else:
        raise ValueError("Unknown data_unit")

    # Hard clip to prevent numeric explosion from loose wires
    eeg_uv = np.clip(eeg_uv, -CLIP_LIMIT_UV, CLIP_LIMIT_UV)
    return eeg_uv / VOLTS_TO_UV

def extract_event_morphology(filtered_eeg, start_sample, end_sample, fs, channel_labels=None, target_points=600):
    segment = filtered_eeg[:, start_sample:end_sample]
    if segment.shape[1] < 10:
        return None

    # Pick the channel with the highest variance (most "active")
    channel_energy = np.var(segment, axis=1)
    max_ch_idx = int(np.argmax(channel_energy))
    best_signal = segment[max_ch_idx, :]

    # Resample for UI visualization
    from scipy.signal import resample
    if len(best_signal) > target_points:
        ui_signal = resample(best_signal, target_points)
        time_axis = np.linspace(0, len(best_signal)/fs, target_points)
    else:
        ui_signal = best_signal
        time_axis = np.linspace(0, len(best_signal)/fs, len(best_signal))

    peak = np.max(np.abs(ui_signal)) + 1e-12
    ui_signal_norm = (ui_signal / peak).tolist()

    ch_label = None
    if channel_labels is not None and max_ch_idx < len(channel_labels):
        ch_label = channel_labels[max_ch_idx]

    return {
        "x": time_axis.tolist(),
        "y": ui_signal_norm,
        "channel_index": max_ch_idx,
        "channel_label": ch_label,
        "peak_amplitude_volts": float(peak)
    }

# --- 3. The Core Processing Logic ---

def process_raw_edf(edf_path):
    """
    Reads EDF, applies Harvard-Standard Filtering, and formats for Model.
    """
    try:
        # Load without preloading first to check channels
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        fs = int(raw.info['sfreq'])

        # --- Channel Selection ---
        eeg_picks = []
        for ch in raw.info['ch_names']:
            try:
                ch_type = raw.get_channel_types(picks=[ch])[0].lower()
            except Exception:
                ch_type = ''
            if ch_type == 'eeg':
                eeg_picks.append(ch)

        if len(eeg_picks) < N_CHANNELS:
            eeg_picks = [ch for ch in raw.info['ch_names'] if 'EEG' in ch.upper() or 'CH' in ch.upper()]

        # Fallback: Just take first N channels if we can't find named EEG channels
        if len(eeg_picks) < 1:
            eeg_picks = raw.info['ch_names'][:N_CHANNELS]
            
        picks = eeg_picks[:N_CHANNELS]
        
        if len(picks) == 0:
            raise ValueError("No valid channels found in EDF.")
            
        raw.pick_channels(picks)
        eeg_data = raw.get_data() # (n_channels, n_samples)

        # 1. Clip artifacts (Safety Net)
        eeg_data = safe_clip_and_unit_convert(eeg_data, data_unit='V')

        # 2. Resampling (Anti-Aliasing Safe)
        # If the file is NOT 256Hz, we must resample it.
        if fs != SAMPLING_RATE:
            # Calculate number of samples
            num_samples = int(eeg_data.shape[1] * SAMPLING_RATE / fs)
            # resample_poly is better than standard resample (implements anti-aliasing filter)
            eeg_data = resample_poly(eeg_data, SAMPLING_RATE, fs, axis=1)
            fs = SAMPLING_RATE

        # 3. Apply Advanced Filtering (Notch + Bandpass)
        filtered_eeg = apply_advanced_filtering(eeg_data, fs)

        # 4. Windowing for Model
        window_samples = int(WINDOW_SECONDS * fs)
        step_samples = int(window_samples * (1 - OVERLAP_PERCENT))
        windows = segment_data_continuous(filtered_eeg, WINDOW_SECONDS, OVERLAP_PERCENT)
        
        if len(windows) == 0:
            return None, None, None

        norm_windows = normalize_windows(windows)
        
        # Reshape for CNN-LSTM: (Batch, Channels, Time, 1)
        model_input = norm_windows.reshape(norm_windows.shape[0], norm_windows.shape[1], norm_windows.shape[2], 1)

        info = {
            'fs': fs,
            'window_samples': window_samples,
            'step_samples': step_samples,
            'ch_names': picks,
            'num_samples': filtered_eeg.shape[1]
        }
        return model_input, filtered_eeg, info

    except Exception as e:
        print(f"Error in process_raw_edf: {e}")
        return None, None, None

# --- 4. Detection Logic (Unchanged) ---
def detect_seizure_with_postprocessing(probability_scores, patient_profile,
                                       ma_window=5, median_k=3,
                                       n_consecutive=3, min_duration_s=6,
                                       peak_req=0.80):
    if len(probability_scores) == 0:
        return {'seizure_detected': False, 'confidence': 0.0, 'clinical_action': 'Monitor', 'evidence': {}}

    smoothed = smooth_probabilities(np.array(probability_scores), ma_window=ma_window, median_k=median_k)

    median_baseline = patient_profile.get('median_baseline', 0.1)
    mad_baseline = patient_profile.get('mad_baseline', 0.05)
    default_thresh = median_baseline + 6 * mad_baseline
    threshold = patient_profile.get('threshold', default_thresh)

    above = (smoothed > threshold).astype(int)
    runs = []
    start = None
    for i, val in enumerate(above):
        if val == 1 and start is None:
            start = i
        elif val == 0 and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(above) - 1))

    events = []
    for (s, e) in runs:
        duration_windows = e - s + 1
        duration_seconds = duration_windows * STEP_SECONDS
        peak_prob = float(np.max(smoothed[s:e+1]))

        if duration_seconds >= min_duration_s and (duration_windows >= n_consecutive or peak_prob >= peak_req):
            events.append({
                'start_window': int(s),
                'end_window': int(e),
                'duration_s': float(duration_seconds),
                'peak_prob': float(peak_prob)
            })

    decision = {
        'seizure_detected': False,
        'confidence': 0.0,
        'clinical_action': 'Monitor as normal',
        'evidence': {
            'threshold_used': float(threshold),
            'smoothed_max': float(np.max(smoothed)) if len(smoothed) > 0 else 0.0,
            'events': events,
            'smoothed_timeline': smoothed.tolist()
        }
    }

    if len(events) >= 1:
        events.sort(key=lambda x: x['peak_prob'], reverse=True)
        e0 = events[0]
        if e0['peak_prob'] >= 0.90:
             decision.update({
                 'seizure_detected': True,
                 'confidence': float(e0['peak_prob']),
                 'clinical_action': 'IMMEDIATE REVIEW - High intensity seizure pattern detected'
             })
        elif e0['duration_s'] >= 15 and e0['peak_prob'] >= 0.80:
             decision.update({
                 'seizure_detected': True,
                 'confidence': float(e0['peak_prob']),
                 'clinical_action': 'IMMEDIATE REVIEW - Sustained seizure activity detected'
             })
        elif e0['peak_prob'] >= 0.80:
             decision.update({
                 'seizure_detected': True,
                 'confidence': float(e0['peak_prob']),
                 'clinical_action': 'REVIEW RECOMMENDED - Moderate confidence'
             })
        else:
             decision.update({
                 'seizure_detected': True,
                 'confidence': 0.60,
                 'clinical_action': 'NOTEWORTHY EVENT - Low confidence, review recommended'
             })

    return decision

# --- 5. Flask Endpoints ---

app = Flask(__name__)

@app.route('/calibrate_multi', methods=['POST'])
def calibrate_multi():
    model = get_model()
    if model is None: return jsonify({'error': 'Model not loaded'}), 500

    if 'files[]' not in request.files or 'patient_id' not in request.form:
        return jsonify({'error': 'Required fields missing'}), 400

    files = request.files.getlist('files[]')
    patient_id = request.form['patient_id']

    all_predictions = []
    for file in files:
        try:
            temp_path = save_uploaded_file_safe(file)
        except Exception as e:
            continue

        try:
            model_input, filtered, info = process_raw_edf(temp_path)
            if model_input is not None:
                preds = model.predict(model_input, verbose=0).flatten()
                all_predictions.extend(preds.tolist())
        except Exception as e:
            print(f"Calibration error: {e}")
        finally:
            try:
                if os.path.exists(temp_path): os.remove(temp_path)
            except Exception: pass

    if len(all_predictions) == 0:
        return jsonify({'error': 'No valid EEG data extracted'}), 500

    arr = np.array(all_predictions)
    arr = np.clip(arr, 0, 1)

    median_baseline = float(np.median(arr))
    mad_baseline = float(mad(arr))

    k = float(request.form.get('mad_k', 6.0))
    new_threshold = min(median_baseline + k * mad_baseline, 0.95)
    if mad_baseline < 1e-4:
        new_threshold = min(np.percentile(arr, 99) + 0.05, 0.95)

    db_path = 'patient_database.json'
    db = {}
    if os.path.exists(db_path):
        try:
            with open(db_path, 'r') as f: db = json.load(f)
        except: pass

    db[patient_id] = {
        'threshold': float(new_threshold),
        'median_baseline': median_baseline,
        'mad_baseline': mad_baseline
    }

    with open(db_path, 'w') as f: json.dump(db, f)

    return jsonify({
        'message': 'Calibration successful',
        'threshold': float(new_threshold),
        'metrics': {'median': median_baseline, 'mad': mad_baseline}
    })

@app.route('/predict_adaptive', methods=['POST'])
def predict_adaptive():
    model = get_model()
    if model is None: return jsonify({'error': 'Model not loaded'}), 500

    if 'file' not in request.files or 'patient_id' not in request.form:
        return jsonify({'error': 'Missing file or patient_id'}), 400

    file = request.files['file']
    patient_id = request.form['patient_id']

    patient_profile = {'threshold': 0.5, 'median_baseline': 0.1, 'mad_baseline': 0.05}
    if os.path.exists('patient_database.json'):
        try:
            with open('patient_database.json', 'r') as f:
                db = json.load(f)
                patient_profile = db.get(patient_id, patient_profile)
        except: pass

    try:
        temp_path = save_uploaded_file_safe(file)
    except Exception as e:
        return jsonify({'error': f'Failed to save uploaded file: {e}'}), 500

    try:
        model_input, filtered_eeg, info = process_raw_edf(temp_path)
        if model_input is None:
            return jsonify({'error': 'Failed to process EDF'}), 500

        raw_preds = model.predict(model_input, verbose=0).flatten()
        probability_scores = np.clip(raw_preds, 0, 1)

        baseline_med = patient_profile.get('median_baseline', 0.1)
        baseline_mad = patient_profile.get('mad_baseline', 0.05)
        z_scores = (probability_scores - baseline_med) / (baseline_mad + 1e-8)

        ma_window = int(request.form.get('ma_window', 5))
        median_k = int(request.form.get('median_k', 3))
        n_consecutive = int(request.form.get('n_consecutive', 3))
        min_duration_s = float(request.form.get('min_duration_s', 6.0))
        peak_req = float(request.form.get('peak_req', 0.80))

        detection_result = detect_seizure_with_postprocessing(
            probability_scores, patient_profile,
            ma_window=ma_window, median_k=median_k,
            n_consecutive=n_consecutive, min_duration_s=min_duration_s,
            peak_req=peak_req
        )

        e = detection_result['evidence']
        events = e.get('events', [])

        if len(events) > 0:
            reported_peak_prob = events[0]['peak_prob']
            reported_duration = events[0]['duration_s']
            start_w = events[0]['start_window']
            end_w = events[0]['end_window']
        else:
            reported_peak_prob = float(np.max(probability_scores)) if len(probability_scores) > 0 else 0.0
            reported_duration = 0.0
            start_w = None
            end_w = None

        # --- UPDATED MORPHOLOGY EXTRACTION (Force QC Logic) ---
        morphology_json = None
        
        # 1. Seizure Case: Use the detected start/end
        if start_w is not None and end_w is not None and filtered_eeg is not None and info is not None:
            fs = info['fs']
            window_samples = info['window_samples']
            step_samples = info['step_samples']
            num_samples = filtered_eeg.shape[1]

            start_sample = max(0, start_w * step_samples)
            end_sample = min(num_samples, end_w * step_samples + window_samples)

            if end_sample - start_sample > 10:
                morphology = extract_event_morphology(filtered_eeg, start_sample, end_sample, fs, channel_labels=info.get('ch_names'))
                if morphology is not None:
                    morphology_json = {
                        'x': morphology['x'],
                        'y': morphology['y'],
                        'channel': morphology.get('channel_label') or f"ch_{morphology.get('channel_index')}",
                        'fs': fs,
                        'original_amplitude_volts': morphology.get('peak_amplitude_volts')
                    }
        
        # 2. NO SEIZURE Case: Force a QC Sample (Middle 3 seconds)
        # This ensures the frontend gets a waveform to display even if normal.
        if morphology_json is None and filtered_eeg is not None and info is not None:
            fs = info['fs']
            num_samples = filtered_eeg.shape[1]
            
            # Define a 3-second window
            qc_duration = 3 * fs
            mid_point = num_samples // 2
            
            start_qc = max(0, mid_point - qc_duration // 2)
            end_qc = min(num_samples, start_qc + qc_duration)
            
            morphology = extract_event_morphology(filtered_eeg, start_qc, end_qc, fs, channel_labels=info.get('ch_names'))
            
            if morphology is not None:
                morphology_json = {
                    'x': morphology['x'],
                    'y': morphology['y'],
                    'channel': morphology.get('channel_label') or "QC_Check",
                    'fs': fs,
                    'original_amplitude_volts': morphology.get('peak_amplitude_volts'),
                    'note': 'QC_SAMPLE_NO_SEIZURE'
                }
        # --------------------------------------------------------

        normalized = {
            'overall_prediction': " SEIZURE DETECTED" if detection_result.get('seizure_detected') else "✓ No Seizure Detected",
            'confidence': detection_result.get('confidence', 0.0),
            'clinical_action': detection_result.get('clinical_action', 'N/A'),
            'windows_analyzed': len(probability_scores),
            'significant_bursts': int(len(events)),
            'peak_probability': reported_peak_prob,
            'detection_span_seconds': reported_duration,
            'high_confidence_windows': int(np.sum(z_scores > 3)),
            'probability_timeline': e.get('smoothed_timeline', probability_scores.tolist()),
            'morphology': morphology_json
        }

        return jsonify(normalized)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        try:
            if os.path.exists(temp_path): os.remove(temp_path)
        except Exception: pass

if __name__ == '__main__':
    app.run(debug=True, port=5000)