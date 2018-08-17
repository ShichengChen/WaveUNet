import numpy as np
quantization_channels=256 #discretize the value to 256 numbers

def mu_law_encode(audio, quantization_channels=256):
    '''Quantizes waveform amplitudes.'''
    mu = (quantization_channels - 1)*1.0
    # Perform mu-law companding transformation (ITU-T, 1988).
    # Minimum operation is here to deal with rare large amplitudes caused
    # by resampling.
    safe_audio_abs = np.minimum(np.abs(audio), 1.0)
    magnitude = np.log1p(mu * safe_audio_abs) / np.log1p(mu)
    signal = np.sign(audio) * magnitude
    # Quantize signal to the specified number of levels.
    return signal
    #return ((signal + 1) / 2 * mu + 0.5).astype(int) #discretize to 0~255



def mu_law_decode(output, quantization_channels=256):
    '''Recovers waveform from quantized values.'''
    mu = quantization_channels - 1
    # Map values back to [-1, 1].
    #signal = 2 * ((output*1.0) / mu) - 1
    signal = output*1.0
    # Perform inverse of mu-law transformation.
    magnitude = (1 / mu) * ((1 + mu)**np.abs(signal) - 1)
    return np.sign(signal) * magnitude

def onehot(a,mu=quantization_channels):
    b = np.zeros((mu,a.shape[0]))
    b[a,np.arange(a.shape[0])] = 1
    return b

def cateToSignal(output, quantization_channels=256,stage=0):
    mu = quantization_channels - 1
    if stage == 0:
        # Map values back to [-1, 1].
        signal = 2 * ((output*1.0) / mu) - 1
        return signal
    else:
        magnitude = (1 / mu) * ((1 + mu)**np.abs(output) - 1)
        return np.sign(output) * magnitude
