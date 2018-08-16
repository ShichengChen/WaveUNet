import h5py
import numpy as np

sampleSize = 16384 * 60
sample_rate = 16384 * 60  # the length of audio for one second

meanx = 0
stdx = 0
meany,meanz = 0,0
stdy,stdz = 0,0
for i in range(50):
    h5f = h5py.File('ccmixter3/'+str(i) + '.h5', 'r')
    x, y, z = h5f['x'][:], h5f['y'][:],h5f['z'][:]
    meanx = x.mean()
    stdx = x.std()
    print(np.sum(x > 1)+np.sum(x < -1))
    exceed = np.sum(((x-meanx) / stdx) > 1)+np.sum(((x-meanx) / stdx) < -1)
    print(exceed)
    print(x.shape)
    print(exceed / x.shape[0])

    #meany = y.mean()
    #stdy = y.std()
    #meanz = z.mean()
    #stdz = z.std()
    h5f.close()

print('xmean,xstd',meanx / 50,stdx / 50)
print('ymean,ystd',meany / 50,stdy / 50)
print('zmean,zstd',meanz / 50,stdz / 50)



# xmean,xstd -0.000824095532298088 0.002741857171058655
# ymean,ystd -0.0008249538391828536 0.0024476730823516845
# zmean,zstd 5.481735206558369e-07 0.0012340296059846878
#
# print(lambda x : x*x)



# factor0 = np.random.uniform(low=0.83, high=1.0)
# factor1 = np.random.uniform(low=0.83, high=1.0)
# #print(factor0,factor1)
# z = z*factor0
# y = y*factor1
# x = (y + z)
#
#
# xmean = x.mean()
# xstd = x.std()
# x = (x - xmean) / xstd
# y = (y - xmean) / xstd