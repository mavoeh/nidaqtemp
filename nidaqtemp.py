import nidaqmx
from nidaqmx import constants
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time

#define function that calculates the temperature
#corresponding to certain voltage
def temp(voltage):
    #constants
    A = 3.9083e-3
    B = -5.775e-7
    #resistance at 0C
    R0 = 1000 
    #resistance at temperature T (formula from latex document)
    Rpt = 6800/(25/voltage -1)
    #temperature
    T = -A/(2*B) - np.sqrt( (Rpt - R0) / (R0 * B) + (A / (2*B))**2 )
    
    return T


class temptask:
    
    def __init__(self, 
                 sampsize = 200,
                 samptime = 100,
                 chanlist = [1,2,3,4,5,6]
                 ):
        """
        initialize the task on the nidaqmax for reading out the voltages
        
        sampsize: number of samples over which the temperature for one measurement is averaged
        samptime: time (in milliseconds!) in which the samples are taken
        chanlist: a list containing the channels to include in this task
        
        from the sample size and time the necessary sampling rate is calculated and passed to the device
        """
        self.chanlist = chanlist
        self.sampsize = sampsize
        self.samptime = samptime
        
        #print(chanlist) #test
        
        #define Task
        self.ai_task = nidaqmx.Task()
        
        #create a string which is passed to the device to specify on which channels to measure
        chanstring = ""
        i = 1
        for chan in chanlist:
            chanstring += "/Dev1/ai" + str(chan-1) +","
        
        # Set terminal_config = constants.TerminalConfiguration.RSE for ground reference!
        self.ai_task.ai_channels.add_ai_voltage_chan(
            physical_channel=chanstring, terminal_config=constants.TerminalConfiguration.RSE
            )
        
        # calculate sample rate from sampsize and samptime
        samprate = int(sampsize/(samptime*1e-3))
        self.ai_task.timing.cfg_samp_clk_timing(
            rate=samprate, samps_per_chan=sampsize, sample_mode=constants.AcquisitionType.FINITE
        )
        
        #get calibration offsets from txt file
        try:
            self.offsets = np.loadtxt("calibration_data.txt")
        except OSError:
            print("Warning: Temperature sensors should be calibrated.")

        
        
    def gettemp(self, channel = False):
        """ 
        If channel = False, reads and returns the temperature from all channels
        If a channel number is specified, return the temperature from just that channel
        """
        
        #write data into array (read method automatically starts task)
        chan_data = np.array(self.ai_task.read(self.sampsize)) #save raw data in numpy array
        
        #in case only 1 chan is measured, still create nested list so that the code still runs properly
        if len(self.chanlist) == 1:
            chan_data = [chan_data]   
        
        #list in which the mean values for the individual channels are saved
        chan_values = np.zeros(len(chan_data))
        
        #take average value of the temperature data for each channel
        #and append it to  chan_values
        for i, single_chan_data in enumerate(chan_data):
            chan_values[i] = single_chan_data.mean()

        #convert voltage values into temperatures
        self.chan_temps = temp(chan_values) - self.offsets[np.array(self.chanlist)-1]
        
        if channel:
            try:
                return self.chan_temps[channel-1]
            except IndexError:
                self.closetask()
                raise ValueError("Please enter a valid channel number.\n")
        else:
            return self.chan_temps
    
    
    
    def calibrate(self, N = 100):
        """
        averages the deviation from the mean temperature of all channels
        over N measurements to find the offset.
        saves the values in the file "calibration_data.txt"
        """
        self.offsets = 0 #remove offsets for re-calibration
        
        t = np.zeros((N,len(self.chanlist))) #array in which the single mesurements are saved
        
        for i in range(N):
            temps = self.gettemp() #measure
            t[i] = temps-temps.mean() #save deviation from mean
            
            #show progress in terminal
            print('Calibrating: {0:_<20s} {1:.0f}%'.format("#"*round(2*i/N*10), round(i/N*100)), end='\r')
        print('Calibrating: {0:_<20s} {1:.0f}%\n\nDone! \
Offsets saved to "calibration_data.txt".\n'.format("#"*20, 100))
        
        mean = np.zeros((1,len(self.chanlist)))
        mean[:] = t.mean(axis=0) #mean deviation for the single channels
        std = t.std(axis=0)   #standard deviation for plot
        
        #save offsets to file
        h = "This file contains the offset values to calibrate the individual temperature sensors:"
        np.savetxt("calibration_data.txt", mean, fmt = "%.3f", delimiter="\t", newline="\n", header = h)


        #create plot which is saved as "calibration.pdf"
        calib_plot = plt.figure()
        plt.errorbar(range(1,7), mean[0], std, linestyle="none",
                     elinewidth = 4, capthick = 4, capsize = 6)
        font = {"fontname":"Times New Roman", "fontsize":18}
        plt.title("Calibration: deviation from mean temperature\n\
averaged over "+str(N)+" measurements", **font)
        plt.xlim(0,7)
        plt.grid(True)
        plt.xticks(range(1,7))
        plt.xlabel("Channel number", **font)
        plt.ylabel(u"Offset from mean temperature (\u00b0C)", **font)
        calib_plot.savefig("calibration.pdf")
        plt.clf()
        
        #load the new offset values
        self.offsets = mean
    
        
        
    def savetemp(self, filename = "temp_record.txt"):
        # saves current tempereture values in the specified file

        #initialize array that is going to be written to file
        temps = np.full(6, np.nan) 
        temps[np.array(self.chanlist)-1] = self.chan_temps
        
        #create the string to write in the file
        filestr = ["{"+str(i)+":.3f}"for i in range(1,7)]
        filestr = "{0:s}\t" + "\t".join(filestr) + "\n"
        
        #get current time to write to file
        now = datetime.now()
        timestr = now.strftime("%d.%m.%Y--%H:%M:%S.%f")[:-5]
        
        #open file in append mode
        with open(filename, "a") as f:
            f.write(filestr.format(timestr, *temps))   
    
    
    
    def record(self, t_total, t_interval = 1, filename = "temp_record.txt"):
        """
        records the temperatures and saves them to a file every
        t_interval seconds for a total time of t_total seconds
        """
        N = int(t_total/t_interval)
        t_start = time.time()
        for i in range(N):
            self.gettemp()
            self.savetemp(filename)
            print('Recording for '+str(t_total)+'s: {0:_<20s} {1:.0f}%'.format(
                  "#"*round(2*i/N*10), round(i/N*100)), end='\r')
            time.sleep( t_interval - ( (time.time() - t_start) % t_interval ) )
        print('Recording for '+str(t_total)+'s: Done! Data saved to "'+str(filename)+'".\n')
        
        
        
    def printtemp(self):
        #prints temperature of all channels in the task
        string = ""
        i = 0
        for ch in self.chanlist:
            string = "ch"+str(ch)+u":  {0:.2f} \u00b0C\t"
            print(string.format(self.chan_temps[i]),end='') #print temp
            i += 1
        print()
        return 0
    
    def closetask(self):
        #has to be called at the end otherwise the device will return an error
        self.ai_task.close()
        return 0
