# WaveUNet
implement Wave-U-Net by pytorch


# Train network
- if you just want to train the model, use commandTrain.py
```
python commandTrain.py --dataset both ##(for both ccmixter and musdb18)
python commandTrain.py --dataset ccmixter ##(for ccmixter only)
python commandTrain.py --dataset musdb18 ##(for musdb18 only)
```
- trainForRandomGen.py (current best result)
- If you want to change the neural network model
- from modelStruct.pyramidnet import Unet[1]
- from modelStruct.unet import Unet[2])
- you can choose one between these two.

# Dataset
- please put mix songs in folder ccmixter2/x
- please put accompaniments in folder ccmixter2/y
- please put vocal songs in folder ccmixter2/z
- *first 50 songs are ccmixter and last 150 songs are musdb18*.
- please name songs as 0.wav, 1.wav, 2.wav etc in folder ccmixter2/x, ccmixter2/y and ccmixter2/z respectively.
- if you only use ccmixter, you should have 0.wav, 2.wav, to 49.wav.
- if you only use musdb18, you should have 50.wav, 51.wav, to 199.wav.
- if you want to use both ccmixter and musdb18, you should have 0.wav, 1.wav, 2.wav, 3.wav, to 199.wav.
- all Audio rates I read are **16000** and **Mono**. 
- I use all ccmixter songs and musdb18 songs, which includes 200 songs.
- training_set = Dataset(np.arange(150), 'ccmixter2/')
- test_set = Testset(np.arange(140,160), 'ccmixter2/')
- validation_set =Valtset(np.arange(150,200), 'ccmixter2/')
- as shown here, I use first 150 songs as training set, last 50 songs as validation set(to visualize loss)
- I will also write results(from 140th songs to 159th songs, which includes training set and validation set) generated from network to folder vsCorpus.

# Installment
 - pytorch 0.4
 - tensorboardX (using tensorboard with pytorch, if you do not want to use tensorboard, set USEBOARD as False)
 - soundfile
 - h5py
 - numpy
# Describe files
## Different start points
 - trainForRandomGen.py (use ccmixter and musdb as dataset to train model)
 - trainchinese.py (use chinese songs as dataset to train model)
 - trainclassify.py (use classification instead regression, classification can also generalize as good as regression but much more noise)
## Tools
 - transformData.py (same as utils file)
## Read Dataset
 - readccmu.py (read ccmixter and musdb18)
 - readchinese.py (read 20000 songs)
 - readpiano.py (read piano songs which is download from youtube to train wavenet, but now it is useless)
## Model structure(all in folder modelStruct)
 - pyramidnet.py(in the middle of nework, use different dilation rate filters to extract features, learned from deep lab series)
 - quanunet.py(use softmax as loss fuction)
 - randomunet.py(my experiment, use random dilation rate, which is inspired by [3])
 - unet.py(use classical wave-u-net[2])
 - unetd.py(use wave-u-net with dilation filters)
 - resunet.py(wanna combine unet and resnet)
 
# Reference
- [1]. https://arxiv.org/pdf/1606.00915.pdf
- [2]. https://qmro.qmul.ac.uk/xmlui/bitstream/handle/123456789/39785/Stoller%20Wave-U-Net%202018%20Accepted.pdf?sequence=1
- [3]. https://arxiv.org/abs/1808.03578
