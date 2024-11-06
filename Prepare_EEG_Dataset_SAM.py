import os
import numpy as np
import pandas as pd 
import pickle
import warnings
import mne
from scipy.io import loadmat
from mne.preprocessing import ICA
from mne_icalabel import label_components
from sklearn.preprocessing import MinMaxScaler

warnings.simplefilter(action="ignore", category=FutureWarning)
mne.set_log_level(verbose = False)
np.random.seed(42)

ch_names = ["Cz", "Fz", "Fp1", "F7", "F3", "FC1", "C3", "FC5", "FT9", "T7", "CP5", "CP1", "P3", 
            "P7", "PO9", "O1", "Pz", "Oz", "O2", "PO10", "P8", "P4", "CP2", "CP6", "T8", "FT10", 
            "FC6", "C4", "FC2", "F4", "F8", "Fp2"]

ch_types = ["eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg",
            "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg",
            "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg",
            "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg"]

sfreq = 128 # raw sampling rate
n_components = 15
scaler = MinMaxScaler()

model_stress = pickle.load(open("stress_model.xgb", "rb"))
stress_features = ["AlphaFp1", "AlphaFp2", "AlphaF3", "AlphaF4", "AlphaF7", "AlphaF8",
       "AlphaFz", "AlphaC3", "AlphaC4", "AlphaCz", "AlphaP3", "AlphaP4",
       "AlphaPz", "AlphaO1", "AlphaO2", "BetaFp1", "BetaFp2", "BetaF3",
       "BetaF4", "BetaF7", "BetaF8", "BetaFz", "BetaC3", "BetaC4", "BetaCz",
       "BetaP3", "BetaP4", "BetaPz", "BetaO1", "BetaO2", "DeltaFp1",
       "DeltaFp2", "DeltaF3", "DeltaF4", "DeltaF7", "DeltaF8", "DeltaFz",
       "DeltaC3", "DeltaC4", "DeltaCz", "DeltaP3", "DeltaP4", "DeltaPz",
       "DeltaO1", "DeltaO2", "ThetaFp1", "ThetaFp2", "ThetaF3", "ThetaF4",
       "ThetaF7", "ThetaF8", "ThetaFz", "ThetaC3", "ThetaC4", "ThetaCz",
       "ThetaP3", "ThetaP4", "ThetaPz", "ThetaO1", "ThetaO2", "GammaFp1",
       "GammaFp2", "GammaF3", "GammaF4", "GammaF7", "GammaF8", "GammaFz",
       "GammaC3", "GammaC4", "GammaCz", "GammaP3", "GammaP4", "GammaPz",
       "GammaO1", "GammaO2"]

model_fatigue = pickle.load(open("fatigue_model.xgb", "rb"))
fatigue_features = ["AlphaFp1", "AlphaFp2", "AlphaC3", "AlphaC4", "AlphaP7", "AlphaP8",
       "AlphaO1", "AlphaO2", "BetaFp1", "BetaFp2", "BetaC3", "BetaC4",
       "BetaP7", "BetaP8", "BetaO1", "BetaO2", "DeltaFp1", "DeltaFp2",
       "DeltaC3", "DeltaC4", "DeltaP7", "DeltaP8", "DeltaO1", "DeltaO2",
       "ThetaFp1", "ThetaFp2", "ThetaC3", "ThetaC4", "ThetaP7", "ThetaP8",
       "ThetaO1", "ThetaO2", "GammaFp1", "GammaFp2", "GammaC3", "GammaC4",
       "GammaP7", "GammaP8", "GammaO1", "GammaO2", "HR", "HRV"]

model_effort = pickle.load(open("effort_model.xgb", "rb"))
model_frustration = pickle.load(open("frustration_model.xgb", "rb"))
model_mental_demand = pickle.load(open("mental_demand_model.xgb", "rb"))
model_physical_demand = pickle.load(open("physical_demand_model.xgb", "rb"))
model_temporal_demand = pickle.load(open("temporal_demand_model.xgb", "rb"))
demand_features = ["AlphaFp1", "AlphaFp2", "AlphaP3", "AlphaP4", "AlphaT7", "AlphaT8",
       "BetaFp1", "BetaFp2", "BetaP3", "BetaP4", "BetaT7", "BetaT8",
       "DeltaFp1", "DeltaFp2", "DeltaP3", "DeltaP4", "DeltaT7", "DeltaT8",
       "ThetaFp1", "ThetaFp2", "ThetaP3", "ThetaP4", "ThetaT7", "ThetaT8",
       "GammaFp1", "GammaFp2", "GammaP3", "GammaP4", "GammaT7", "GammaT8", "HR", "HRV"]

def normalize(dataframe):
    column_names_to_normalize = dataframe.columns[:95]
    x = dataframe[column_names_to_normalize].values
    x_scaled = scaler.fit_transform(x)
    df_temp = pd.DataFrame(x_scaled, columns=column_names_to_normalize, index = dataframe.index)
    dataframe[column_names_to_normalize] = df_temp
    return dataframe

def psd_frontal(epoch_whole, frequency, low, high):
    freq_bands = {frequency: [low, high]}
    spectrum = epoch_whole.compute_psd(picks = ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"], fmin = 0.5, fmax = 45)
    psds, freqs = spectrum.get_data(return_freqs = True)
    X = []
    for fmin, fmax in freq_bands.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis = -1)
        X.append(psds_band.reshape(len(psds), -1))
    whole_psd = np.concatenate(X, axis = 1)
    whole_psd = np.multiply(whole_psd, 1e12)    
    powers = pd.DataFrame(whole_psd, columns = [frequency + "Fp1", frequency + "Fp2", frequency + "F3", frequency + "F4", frequency + "F7", frequency + "F8", frequency + "Fz"])
    return powers

def psd_central(epoch_whole, frequency, low, high):
    freq_bands = {frequency: [low, high]}
    spectrum = epoch_whole.compute_psd(picks = ["C3", "C4", "Cz"], fmin = 0.5, fmax = 45)
    psds, freqs = spectrum.get_data(return_freqs = True)
    X = []
    for fmin, fmax in freq_bands.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis = -1)
        X.append(psds_band.reshape(len(psds), -1))
    whole_psd = np.concatenate(X, axis = 1)
    whole_psd = np.multiply(whole_psd, 1e12)    
    powers = pd.DataFrame(whole_psd, columns = [frequency + "C3", frequency + "C4", frequency + "Cz"])
    return powers

def psd_parietal(epoch_whole, frequency, low, high):
    freq_bands = {frequency: [low, high]}
    spectrum = epoch_whole.compute_psd(picks = ["P3", "P4", "P7", "P8", "Pz"], fmin = 0.5, fmax = 45)
    psds, freqs = spectrum.get_data(return_freqs = True)
    X = []
    for fmin, fmax in freq_bands.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis = -1)
        X.append(psds_band.reshape(len(psds), -1))
    whole_psd = np.concatenate(X, axis = 1)
    whole_psd = np.multiply(whole_psd, 1e12)    
    powers = pd.DataFrame(whole_psd, columns = [frequency + "P3", frequency + "P4", frequency + "P7", frequency + "P8", frequency + "Pz"])
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

final_dataset = pd.DataFrame()
for subject in range(1,41):
    print("Processing: " + str(subject))
    # Relax
    for trial in range(1,4):
        temp = loadmat(os.path.join("SAM", "Relax_sub_" + str(subject) + "_trial" + str(trial) + ".mat"), simplify_cells = True)
        df = pd.DataFrame(temp["Data"].T, columns = ch_names)
        info = mne.create_info(ch_names=ch_names, sfreq=128, ch_types=ch_types)
        samples = df.T / 1e8
        data_raw = mne.io.RawArray(samples, info)
        montage = mne.channels.make_standard_montage('standard_1020')
        data_raw.set_montage(montage)
        # reference
        data_raw.set_eeg_reference(ref_channels = "average")
        # filter
        data_filtered = data_raw.copy().filter(l_freq=1.0, h_freq=60.0)
        # ICA
        ica = ICA(n_components = n_components, method = "infomax", random_state = 42, fit_params=dict(extended=True))
        ica.fit(data_filtered)
        ic_labels = label_components(data_filtered, ica, method="iclabel")
        labels = ic_labels["labels"]
        exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
        reconstructed = data_filtered.copy()
        ica.apply(reconstructed, exclude=exclude_idx)
        data_filtered = reconstructed.filter(l_freq=1, h_freq=45)
        data_filtered = data_filtered.resample(sfreq = 250)
        epochs = mne.make_fixed_length_epochs(data_filtered, duration=10, preload=True)
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
            subset["Stress"] = model_stress.predict(subset[stress_features])[0]
            subset["Subject"] = str(subject)
            subset["Index"] = str(epoch)
            subset["Task"] = "Relax"
            subset["Trial"] = str(trial)
            final_dataset = pd.concat([final_dataset, subset], axis=0 ,ignore_index = True)

    # Mirror
    for trial in range(1,4):
        temp = loadmat(os.path.join("SAM", "Mirror_image_sub_" + str(subject) + "_trial" + str(trial) + ".mat"), simplify_cells = True)
        df = pd.DataFrame(temp["Data"].T, columns = ch_names)
        info = mne.create_info(ch_names=ch_names, sfreq=128, ch_types=ch_types)
        samples = df.T / 1e8
        data_raw = mne.io.RawArray(samples, info)
        montage = mne.channels.make_standard_montage('standard_1020')
        data_raw.set_montage(montage)
        # reference
        data_raw.set_eeg_reference(ref_channels = "average")
        # filter
        data_filtered = data_raw.copy().filter(l_freq=1.0, h_freq=60.0)
        # ICA
        ica = ICA(n_components = n_components, method = "infomax", random_state = 42, fit_params=dict(extended=True))
        ica.fit(data_filtered)
        ic_labels = label_components(data_filtered, ica, method="iclabel")
        labels = ic_labels["labels"]
        exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
        reconstructed = data_filtered.copy()
        ica.apply(reconstructed, exclude=exclude_idx)
        data_filtered = reconstructed.filter(l_freq=1, h_freq=45)
        data_filtered = data_filtered.resample(sfreq = 250)
        epochs = mne.make_fixed_length_epochs(data_filtered, duration=10, preload=True)
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
            subset["Stress"] = model_stress.predict(subset[stress_features])[0]
            subset["Subject"] = str(subject)
            subset["Index"] = str(epoch)
            subset["Task"] = "Mirror"
            subset["Trial"] = str(trial)
            final_dataset = pd.concat([final_dataset, subset], axis=0 ,ignore_index = True)

    # Arithmetic
    for trial in range(1,4):
        temp = loadmat(os.path.join("SAM", "Arithmetic_sub_" + str(subject) + "_trial" + str(trial) + ".mat"), simplify_cells = True)
        df = pd.DataFrame(temp["Data"].T, columns = ch_names)
        info = mne.create_info(ch_names=ch_names, sfreq=128, ch_types=ch_types)
        samples = df.T / 1e8
        data_raw = mne.io.RawArray(samples, info)
        montage = mne.channels.make_standard_montage('standard_1020')
        data_raw.set_montage(montage)
        # reference
        data_raw.set_eeg_reference(ref_channels = "average")
        # filter
        data_filtered = data_raw.copy().filter(l_freq=1.0, h_freq=60.0)
        # ICA
        ica = ICA(n_components = n_components, method = "infomax", random_state = 42, fit_params=dict(extended=True))
        ica.fit(data_filtered)
        ic_labels = label_components(data_filtered, ica, method="iclabel")
        labels = ic_labels["labels"]
        exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
        reconstructed = data_filtered.copy()
        ica.apply(reconstructed, exclude=exclude_idx)
        data_filtered = reconstructed.filter(l_freq=1, h_freq=45)
        data_filtered = data_filtered.resample(sfreq = 250)
        epochs = mne.make_fixed_length_epochs(data_filtered, duration=10, preload=True)
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
            subset["Stress"] = model_stress.predict(subset[stress_features])[0]
            subset["Subject"] = str(subject)
            subset["Index"] = str(epoch)
            subset["Task"] = "Arithmetic"
            subset["Trial"] = str(trial)
            final_dataset = pd.concat([final_dataset, subset], axis=0 ,ignore_index = True)

    # Stroop
    for trial in range(1,4):
        temp = loadmat(os.path.join("SAM", "Stroop_sub_" + str(subject) + "_trial" + str(trial) + ".mat"), simplify_cells = True)
        df = pd.DataFrame(temp["Data"].T, columns = ch_names)
        info = mne.create_info(ch_names=ch_names, sfreq=128, ch_types=ch_types)
        samples = df.T / 1e8
        data_raw = mne.io.RawArray(samples, info)
        montage = mne.channels.make_standard_montage('standard_1020')
        data_raw.set_montage(montage)
        # reference
        data_raw.set_eeg_reference(ref_channels = "average")
        # filter
        data_filtered = data_raw.copy().filter(l_freq=1.0, h_freq=60.0)
        # ICA
        ica = ICA(n_components = n_components, method = "infomax", random_state = 42, fit_params=dict(extended=True))
        ica.fit(data_filtered)
        ic_labels = label_components(data_filtered, ica, method="iclabel")
        labels = ic_labels["labels"]
        exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
        reconstructed = data_filtered.copy()
        ica.apply(reconstructed, exclude=exclude_idx)
        data_filtered = reconstructed.filter(l_freq=1, h_freq=45)
        data_filtered = data_filtered.resample(sfreq = 250)
        epochs = mne.make_fixed_length_epochs(data_filtered, duration=10, preload=True)
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
            subset["Stress"] = model_stress.predict(subset[stress_features])[0]
            subset["Subject"] = str(subject)
            subset["Index"] = str(epoch)
            subset["Task"] = "Stroop"
            subset["Trial"] = str(trial)
            final_dataset = pd.concat([final_dataset, subset], axis=0 ,ignore_index = True)

final_dataset = normalize(final_dataset)   
final_dataset.to_csv("EEG_Dataset_SAM.csv", index=False)
