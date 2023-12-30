from IPython.display import Image, Audio
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
from scipy.signal import find_peaks
import urllib.request
%matplotlib inline


# obtain/fetch audio file from the Internet
url_nasa = "https://www.googleapis.com/drive/v3/files/18A7xp7aeslDjxr2iXUzSWXJEcB2n1uUl?alt=media&key=AIzaSyD4hbzzR4ie-y2nDl8Yd9B9cnss5dZIWGU"
local_filename, headers = urllib.request.urlretrieve(url_nasa)
fs, x = scipy.io.wavfile.read(local_filename) 
if len(x.shape) > 1:
    x = x[:, 0]

t = np.linspace(0, len(x)/fs, len(x), endpoint=False)


total = int(len(x))
minutes = int((len(x)/fs)/60)
interval = int(60*fs)
i = 0
time = 1

for i in range(0, total, interval):
  if i + interval < total:
    start = i
    end = i+interval
    a = x[start : end]
    b = np.linspace(start, 60*time, int(len(x)/minutes), endpoint=False)
    time = time+1
    peaks, _ = find_peaks(a, height=2000, distance=4000)
    print(len(peaks))
    if time==3:
      fig1, ax1 = plt.subplots()
      fig2, ax2 = plt.subplots()
      ax1.plot(a, label='time')
      ax2.plot(peaks, a[peaks], "x")
      



#print(total)

#peaks, _ = find_peaks(x, height=3000, distance=4000, )

#fig1, ax1 = plt.subplots()
#fig2, ax2 = plt.subplots()
#ax1.plot(t, x, label='time')
#ax2.plot(peaks, x[peaks], "x")

#ax2.plot(np.zeros_like(x), "--", color="gray")

#np.diff(peaks)
#print(len(peaks))


