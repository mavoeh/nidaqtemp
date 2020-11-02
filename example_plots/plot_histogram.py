import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so

font = {"fontname":"Times New Roman", "fontsize":22}


sampsize = 200
samptime = 100

t, T = np.loadtxt('sampsize'+str(sampsize)+'_samptime'+str(samptime)+'.txt', unpack = True)


dist = t[1:]-t[:-1]
print("Mean distance between samples: {0:.4f} +/- {1:.4f} s  ({2:.2f} % std)".format(dist.mean(),dist.std(),100*dist.std()/dist.mean()))
f = 1/dist.mean() #sampling frequency

ft = np.fft.fft(T)
freqs = np.fft.fftfreq(len(T))  #frequencies of the fourier transform components
freqs = freqs * f

low_freqs = freqs[(freqs>=0)&(freqs<0.03)]
cut = len(low_freqs)
print(cut)
print("Highest frequency cut: {0:.3f} mHz, lowest frequency uncut: {1:.3f} mHz".format(freqs[cut-1]*1e3,freqs[cut]*1e3))
ft[:cut] = 0
ft[-cut:] = 0
T_filtered = np.fft.ifft(ft)

fig1 = plt.figure()

plt.plot(t, T_filtered,label="filtered")
plt.plot(t, T-T.mean(),alpha=0.7,label="unfiltered")

plt.ylabel("Temperature deviation form mean value ($^\circ$C)", **font)
plt.xlabel("Time (s)", **font)
plt.legend(loc=0)

#fig1.savefig("figures/sampsize"+str(sampsize)+"samptime"+str(samptime)+"_data.pdf")



fig2 = plt.figure()

y, bins, _ = plt.hist(np.real(T_filtered),normed=True,bins=20,label="filtered")

plt.hist(T-T.mean(),bins=20,alpha=0.4,edgecolor='none',label="unfiltered",normed=True)

x = bins[1:]-(bins[1]-bins[0])/2

def gauss(x, mu, sig):
    return 1/np.sqrt(2*np.pi*sig**2) * np.exp(-(x-mu)**2/(2*sig**2))

popt, pcov = so.curve_fit(gauss, x, y)
perr = np.sqrt(np.diag(pcov))

minval = min(np.real(T_filtered))
maxval = max(np.real(T_filtered))
xarr = np.linspace(minval,maxval,300)
plt.plot(xarr, gauss(xarr, *popt), c="red",lw=3,label="gaussian fit")

amp = 1/np.sqrt(2*np.pi*popt[1]**2)
        
plt.plot([], [], ' ',
        label="$\mu$ = ({0:.2f}$\pm${1:.2f})mK\n$\sigma$ = ({2:.2f}$\pm${3:.2f})mK"\
        .format(popt[0]*1e3,perr[0]*1e3,popt[1]*1e3,perr[1]*1e3))

plt.xlabel("Temperature around mean value ($^\circ$C)", **font)
plt.ylabel("Probability distribution", **font)
plt.legend(loc=0)
#plt.title("Probability distribution of Temperatures for\n samplesize = "+str(sampsize)+" and sample time = "+str(samptime)+" ms", **font)

print("\n\nFit results for gaussian fit:\n")
print("mu    = {0:.5f} +/- {1:.5f}\nsigma = {2:.5f} +/- {3:.5f}".format(popt[0],perr[0],popt[1],perr[1]))

#fig2.savefig("figures/sampsize"+str(sampsize)+"samptime"+str(samptime)+"_hist.pdf")

plt.show()
