from mir_eval.separation import bss_eval_sources
import soundfile as sf


#x, _ = sf.read('/home/ubuntu/music2/chinesesongs/0/0.wav')
#y, _ = sf.read('/home/ubuntu/music2/chinesesongs/0/1.wav')
#print(bss_eval_sources(x, y))

#x, _ = sf.read('/home/ubuntu/music2/chinesesongs/1/0.wav')
#y, _ = sf.read('/home/ubuntu/music2/chinesesongs/1/1.wav')
#print(bss_eval_sources(x, y))

x, _ = sf.read('/home/ubuntu/music2/chinesesongs/2/0.wav')
y, _ = sf.read('/home/ubuntu/music2/chinesesongs/2/1.wav')
nmax = -1000
nind = 0
for i in range(-32000,-1,100):
    sdr = bss_eval_sources(x[32000+i:i], y[32000:])
    print(i,sdr)
    if(nmax <= sdr):
        nmax = sdr
        nind = i
for i in range(1,32000,100):
    sdr = bss_eval_sources(x[32000+i:], y[32000:-i])
    print(i, sdr)
    if (nmax <= sdr):
        nmax = sdr
        nind = i


#x, _ = sf.read('/home/ubuntu/music2/chinesesongs/2/0.wav')
#y, _ = sf.read('/home/ubuntu/music2/chinesesongs/2/0.wav')
#print(bss_eval_sources(x, y))