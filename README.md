# WaveUNet
implement Wave-U-Net by pytorch


# Train network
- trainForRandomGen.py (current best result)
- you can change the neural network model
- from modelStruct.pyramidnet import [Unet](https://arxiv.org/pdf/1606.00915.pdf)
- from modelStruct.unet import [Unet](https://qmro.qmul.ac.uk/xmlui/bitstream/handle/123456789/39785/Stoller%20Wave-U-Net%202018%20Accepted.pdf?sequence=1)
- you can choose one between these two.

# Dataset
- please put mix songs in folder ccmixter2/x
- please put accompaniments in folder ccmixter2/y
- please put vocal songs in folder ccmixter2/z
- please name songs as 0.wav, 1.wav, 2.wav etc in folder ccmixter2/x, ccmixter2/y and ccmixter2/z respectively.
- all Audio rates are *16000* and *Mono* instead of stereo 
- I use all ccmixter songs and musdb18 songs, which includes 200 songs.
- training_set = Dataset(np.arange(150), 'ccmixter2/')
- test_set = Testset(np.arange(140,160), 'ccmixter2/')
- validation_set =Valtset(np.arange(150,200), 'ccmixter2/')
- as shown here, I use first 150 songs as training set, last 50 songs as validation set(to visualize loss)
- I will also write results(from 140th songs to 159th songs, which includes training set and validation set) generated from network to folder vsCorpus.

# installment
 - pytorch 0.4
 - tensorboardX (using tensorboard with pytorch, if you do not want to use tensorboard, set USEBOARD as False)
 - soundfile
 - h5py
 - numpy
