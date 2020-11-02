import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so

font = {"fontname":"Times New Roman", "fontsize":18}

sampsizes = [10,15,20,25,30,35,40,50,75,100,125,150,175,200,250,300,400,500]
#temps = []
stds = []
stds_filtered = []
fts = []
st = 100
for ss in sampsizes:
    time, temp = np.loadtxt("sampsize"+str(ss)+"_samptime"+str(st)+".txt", unpack = True)
    #temps.append(temp)
    stds.append(temp.std())
    
    dist = time[1:]-time[:-1]
    print("Mean distance between samples: {0:.4f} +/- {1:.4f} s  ({2:.2f} % std)".format(dist.mean(),dist.std(),100*dist.std()/dist.mean()))
    f = 1/dist.mean() #sampling frequency
    
    ft = np.fft.fft(temp)
    fts.append(ft)
    freqs = np.fft.fftfreq(len(temp))  #frequencies of the fourier transform components
    freqs = freqs * f
    
    
    #cut = 10
    low_freqs = freqs[(freqs>=0)&(freqs<0.03)]
    cut = len(low_freqs)
    ft[:cut] = 0
    ft[-cut:] = 0
    temp_filtered = np.fft.ifft(ft)
    

    if ss==200:
        plt.figure()
        plt.plot(time[10*cut:-10*cut], np.real(temp_filtered[10*cut:-10*cut]),label="filtered")
        plt.plot(time[10*cut:-10*cut], np.imag(temp_filtered[10*cut:-10*cut]),label="filtered im",lw=5)
        plt.plot(time[10*cut:-10*cut], temp[10*cut:-10*cut] - np.mean(temp[10*cut:-10*cut]), label="unfiltered",alpha=0.5)
        plt.legend(loc=0)
        plt.title("Temperature recording for samplesize = "+str(ss)+"\nand samplingtime = "+str(st)+" s", **font)
        plt.xlabel("Time (s)", fontsize=16)
        plt.ylabel("Temperature ($^\circ$C)", fontsize=16)
        
        fig_fourier = plt.figure()
        plt.plot(freqs[freqs>=0],np.abs(ft)[freqs>=0],c="g")
        plt.plot(freqs[freqs<0],np.abs(ft)[freqs<0],c="g")
        plt.xlabel("Frequency (Hz)", **font)
        plt.ylabel("Fourier component ($^\circ$C/Hz)", **font)
        #plt.title("Fourier transform of temperature recording", **font)
        
        #fig_fourier.savefig("figures/fourier.pdf")

    
    stds_filtered.append(temp_filtered[10*cut:-10*cut].std())
print("\n\n\n")

def fit(x, a, b, c):
    return a*x**b + c


popt,pcov = so.curve_fit(fit, sampsizes[:-1], stds_filtered[:-1], p0 = [1.,-1,0])#, sigma = sig, p0 = [1.,-1,0])
perr = np.sqrt(np.diag(pcov))

print("Fit results:\n")

print("f(s) = a * x^b + c\n")

print("a = {0:.2f} +/- {1:.2f}".format(popt[0],perr[0]))
print("b = {0:.2f} +/- {1:.2f}".format(popt[1],perr[1]))
print("c = {0:.3f} +/- {1:.3f}".format(popt[2],perr[2]))


fig2 = plt.figure()

plt.scatter(sampsizes,stds,label="unfiltered",c="red")
plt.scatter(sampsizes,stds_filtered,label="filtered",c="blue")

xarr = np.linspace(sampsizes[0],sampsizes[-1],300)
plt.plot(xarr, fit(xarr, *popt), c = "blue", label="fit")

plt.xlabel("Sample size", **font)
plt.ylabel(r"Standard deviation ($^\circ$C)", **font)

plt.xlim(0,510)
plt.ylim(0,0.15)
plt.legend(loc=0)
plt.grid(True)


#fig2.savefig("figures/sampsize_characterization.pdf")

plt.show()

