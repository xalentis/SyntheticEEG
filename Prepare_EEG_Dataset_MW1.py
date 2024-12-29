import os
import numpy as np
import pandas as pd 
import pickle
import warnings
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
from sklearn.preprocessing import MinMaxScaler
import xgboost

warnings.simplefilter(action="ignore", category=FutureWarning)
mne.set_log_level(verbose = False)
np.random.seed(42)

ch_names = ["AF3", "AF4", "F3", "F4", "F7", "F8", "FC5",
            "FC6", "P7", "P8", "T7", "T8", "O1", "O2"]

ch_types = ["eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", 
            "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg",]

sfreq = 128 # raw sampling rate
n_components = 10
scaler = MinMaxScaler()


model_stress = xgboost.XGBClassifier()
model_stress.load_model("stress_simple_model.xgb")
stress_features = ["HR", "HRV"]

def upsample(lst, target_length):
    o_indices = np.linspace(0, len(lst) - 1, num=len(lst))
    t_indices = np.linspace(0, len(lst) - 1, num=target_length)
    upsampled = np.interp(t_indices, o_indices, lst)
    return upsampled.tolist()

def downsample(lst, target_length):
    step = len(lst) / target_length
    downsampled = []
    for i in range(target_length):
        start = int(i * step)
        end = int((i + 1) * step)
        chunk = lst[start:end]
        chunk_mean = sum(chunk) / len(chunk) if chunk else 0
        downsampled.append(chunk_mean)
    return downsampled

def normalize(dataframe):
    column_names_to_normalize = dataframe.columns[:70]
    x = dataframe[column_names_to_normalize].values
    x_scaled = scaler.fit_transform(x)
    df_temp = pd.DataFrame(x_scaled, columns=column_names_to_normalize, index = dataframe.index)
    dataframe[column_names_to_normalize] = df_temp
    return dataframe

def psd_frontal(epoch_whole, frequency, low, high):
    freq_bands = {frequency: [low, high]}
    spectrum = epoch_whole.compute_psd(picks = ["AF3", "AF4", "F3", "F4", "F7", "F8"], fmin = 0.5, fmax = 45)
    psds, freqs = spectrum.get_data(return_freqs = True)
    X = []
    for fmin, fmax in freq_bands.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis = -1)
        X.append(psds_band.reshape(len(psds), -1))
    whole_psd = np.concatenate(X, axis = 1)
    whole_psd = np.multiply(whole_psd, 1e12)    
    powers = pd.DataFrame(whole_psd, columns = [frequency + "AF3", frequency + "AF4", frequency + "F3", frequency + "F4", frequency + "F7", frequency + "F8"])
    return powers

def psd_central(epoch_whole, frequency, low, high):
    freq_bands = {frequency: [low, high]}
    spectrum = epoch_whole.compute_psd(picks = ["FC5", "FC6"], fmin = 0.5, fmax = 45)
    psds, freqs = spectrum.get_data(return_freqs = True)
    X = []
    for fmin, fmax in freq_bands.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis = -1)
        X.append(psds_band.reshape(len(psds), -1))
    whole_psd = np.concatenate(X, axis = 1)
    whole_psd = np.multiply(whole_psd, 1e12)    
    powers = pd.DataFrame(whole_psd, columns = [frequency + "FC5", frequency + "FC6"])
    return powers

def psd_parietal(epoch_whole, frequency, low, high):
    freq_bands = {frequency: [low, high]}
    spectrum = epoch_whole.compute_psd(picks = ["P7", "P8"], fmin = 0.5, fmax = 45)
    psds, freqs = spectrum.get_data(return_freqs = True)
    X = []
    for fmin, fmax in freq_bands.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis = -1)
        X.append(psds_band.reshape(len(psds), -1))
    whole_psd = np.concatenate(X, axis = 1)
    whole_psd = np.multiply(whole_psd, 1e12)    
    powers = pd.DataFrame(whole_psd, columns = [frequency + "P7", frequency + "P8"])
    return powers

def psd_occipital(epoch_whole, frequency, low, high):
    freq_bands = {frequency: [low, high]}
    spectrum = epoch_whole.compute_psd(picks = ["O1", "O2"], fmin = 0.5, fmax = 45)
    psds, freqs = spectrum.get_data(return_freqs = True)
    X = []
    for fmin, fmax in freq_bands.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis = -1)
        X.append(psds_band.reshape(len(psds), -1))
    whole_psd = np.concatenate(X, axis = 1)
    whole_psd = np.multiply(whole_psd, 1e12)    
    powers = pd.DataFrame(whole_psd, columns = [frequency + "O1", frequency + "O2"])
    return powers

def psd_temporal(epoch_whole, frequency, low, high):
    freq_bands = {frequency: [low, high]}
    spectrum = epoch_whole.compute_psd(picks = ["T7", "T8"], fmin = 0.5, fmax = 45)
    psds, freqs = spectrum.get_data(return_freqs = True)
    X = []
    for fmin, fmax in freq_bands.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis = -1)
        X.append(psds_band.reshape(len(psds), -1))
    whole_psd = np.concatenate(X, axis = 1)
    whole_psd = np.multiply(whole_psd, 1e12)    
    powers = pd.DataFrame(whole_psd, columns = [frequency + "T7", frequency + "T8"])
    return powers

def calculate_rolling_rmssd(ibi_list, sampling_rate=0.5, window_size=10, step_size=10):
    ibi_seconds = np.array(ibi_list) / 1000
    window_samples = int(window_size * sampling_rate)
    step_samples = int(step_size * sampling_rate)
    rolling_rmssd = []
    for start in range(0, len(ibi_seconds) - window_samples + 1, step_samples):
        window = ibi_seconds[start:start + window_samples]
        successive_diffs = np.diff(window)
        rmssd = np.sqrt(np.mean(successive_diffs ** 2))
        rolling_rmssd.append(rmssd)
    return rolling_rmssd

final_dataset = pd.DataFrame()

subjects = ["01", "02", "03", "04", "05", "06", "07", "08", "10", "11", "12", "13", "15"] # 9 and 14 no HR data
scores_data = pd.read_parquet(os.path.join("MentalWorkloadEEG", "data_n_back_test", "game_performance", "game_scores.parquet"), engine="pyarrow")
hr_data = pd.read_parquet(os.path.join("MentalWorkloadEEG", "data_n_back_test", "ecg", "ecg_hr.parquet"), engine="pyarrow")
ibi_data = pd.read_parquet(os.path.join("MentalWorkloadEEG", "data_n_back_test", "ecg", "ecg_ibi.parquet"), engine="pyarrow")
eeg_data = pd.read_parquet(os.path.join("MentalWorkloadEEG", "data_n_back_test", "eeg", "eeg.parquet"), engine="pyarrow")

for subject in subjects:
    print("Processing: " + subject)
    for test in range(1,4):
        score_mean = np.mean(scores_data.loc[(scores_data["subject"] == "subject_" + subject) & (scores_data["test"] == str(test))]["score"].values[0])
        hr = hr_data.loc[(hr_data["subject"] == "subject_" + subject) & (hr_data["test"] == test)]
        hr = hr["hr"].values
        ibi = ibi_data.loc[(ibi_data["subject"] == "subject_" + subject) & (ibi_data["test"] == test)]
        ibi = ibi["rr_int"].values.tolist()
        hrv = calculate_rolling_rmssd(ibi)
        subset = eeg_data.loc[(eeg_data["subject"] == "subject_" + subject) & (eeg_data["test"] == test)]
        subset = subset[["EEG.AF3", "EEG.AF4", "EEG.F3", "EEG.F4", "EEG.F7", "EEG.F8", 
                        "EEG.FC5", "EEG.FC6", "EEG.P7", "EEG.P8", "EEG.T7", "EEG.T8", "EEG.O1", "EEG.O2"]]
        subset = pd.DataFrame.transpose(subset)
        info = mne.create_info(ch_names = ch_names, ch_types = ch_types, sfreq = sfreq)
        data_raw = mne.io.RawArray(subset, info) # create mne-class array  
        montage = mne.channels.make_standard_montage("standard_1020")
        data_raw.set_montage(montage)
        # reference
        data_raw.set_eeg_reference(ref_channels = "average")
        # filter
        data_filtered = data_raw.filter(l_freq=1, h_freq=45)
        data_filtered = data_filtered.resample(sfreq = 250)
        epochs = mne.make_fixed_length_epochs(data_filtered, duration=10, preload=True)
        HR = upsample(hr, len(epochs))
        HRV = downsample(hrv, len(epochs))
        for epoch in range(0, len(epochs)):
            # psd
            alpha_frontal = psd_frontal(epochs[epoch], "Alpha", 8, 12)
            alpha_central = psd_central(epochs[epoch], "Alpha", 8, 12)
            alpha_parietal = psd_parietal(epochs[epoch], "Alpha", 8, 12)
            alpha_occipital = psd_occipital(epochs[epoch], "Alpha", 8, 12)
            alpha_temporal = psd_temporal(epochs[epoch], "Alpha", 8, 12)
            beta_frontal = psd_frontal(epochs[epoch], "Beta", 12, 30)
            beta_central = psd_central(epochs[epoch], "Beta", 12, 30)
            beta_parietal = psd_parietal(epochs[epoch], "Beta", 12, 30)
            beta_occipital = psd_occipital(epochs[epoch], "Beta", 12, 30)
            beta_temporal = psd_temporal(epochs[epoch], "Beta", 8, 12)
            delta_frontal = psd_frontal(epochs[epoch], "Delta", 0.5, 4)
            delta_central = psd_central(epochs[epoch], "Delta", 0.5, 4)
            delta_parietal = psd_parietal(epochs[epoch], "Delta", 0.5, 4)
            delta_occipital = psd_occipital(epochs[epoch], "Delta", 0.5, 4)
            delta_temporal = psd_temporal(epochs[epoch], "Delta", 8, 12)
            theta_frontal = psd_frontal(epochs[epoch], "Theta", 4, 8)
            theta_central = psd_central(epochs[epoch], "Theta", 4, 8)
            theta_parietal = psd_parietal(epochs[epoch], "Theta", 4, 8)
            theta_occipital = psd_occipital(epochs[epoch], "Theta", 4, 8)
            theta_temporal = psd_temporal(epochs[epoch], "Theta", 8, 12)
            gamma_frontal = psd_frontal(epochs[epoch], "Gamma", 30, 45)
            gamma_central = psd_central(epochs[epoch], "Gamma", 30, 45)
            gamma_parietal = psd_parietal(epochs[epoch], "Gamma", 30, 45)
            gamma_occipital = psd_occipital(epochs[epoch], "Gamma", 30, 45)
            gamma_temporal = psd_temporal(epochs[epoch], "Gamma", 8, 12)
            subset = pd.concat([alpha_frontal, alpha_central, alpha_parietal, alpha_occipital, alpha_temporal, \
                                beta_frontal, beta_central, beta_parietal, beta_occipital, beta_temporal, \
                                delta_frontal, delta_central, delta_parietal, delta_occipital, delta_temporal, \
                                theta_frontal, theta_central, theta_parietal, theta_occipital, theta_temporal, \
                                gamma_frontal, gamma_central, gamma_parietal, gamma_occipital, gamma_temporal], axis=1)
            subset["HR"] = HR[epoch]
            subset["HRV"] = HRV[epoch]
            subset["Subject"] = str(subject)
            subset["Test"] = str(test)
            subset["Score"] = str(score_mean)
            subset["Stress"] = model_stress.predict_proba(subset[stress_features])[0][1]
            subset["Index"] = str(epoch)
            final_dataset = pd.concat([final_dataset, subset], axis=0 ,ignore_index = True)

final_dataset = normalize(final_dataset)   
final_dataset.to_csv("EEG_Dataset_MW1.csv", index=False)