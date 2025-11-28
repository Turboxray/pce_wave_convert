import argparse
import os
import numpy
import sys
import resampy
import pathlib
import tkthread; tkthread.patch()   # do this before importing tkinter

from functools import partial

import threading

from tkinter import filedialog as fd
import tkinter as tk
from tkinter import ttk

import simpleaudio as sa

from numba import jit

import matplotlib.pyplot as plt

HEXbase = 16
DECbase = 10
INT_MAX = 2**32 - 1
INT_MIN = (INT_MAX) / -2 - 1

runOptions = {}
components = {}
filterValues    = ['kaiser_best','kaiser_fast']
bitdepthValues  = ['8bit']
playerateValues = ['6960','6991','7020','9279','10440','11090','12180','13500','13920','14040','15360','15740','22020','31480','no resample']

@jit(nopython=True)
def convertFreq(data, newFreq, currFreq, expand5bit=False, expand4bit=False, vol_adjust=1.0):

    # want this to sound crunch-y so using nearest neighbor
    offset = ((255-(255*vol_adjust))/2)
    print(f"vol adjust: {vol_adjust}")
    steps = newFreq / currFreq
    preInc = 0.0
    currInc = steps
    output = []
    if expand4bit:
        shift = 4
    elif expand5bit:
        shift = 3
    else:
        shift = 0
    for sample in data:
        delta = int(currInc) - int(preInc)
        for _ in range(delta):
            output.append( int( ((sample<<shift)*vol_adjust) + offset) )
        preInc = currInc
        currInc += steps

    return output

class ConvertWave():

    def __init__(self):
        pass

    def convertPCMData(self, sampleObj, sampleData, ):

        sr_orig         = sampleObj['SamplesPerSec']
        playbackRate    = sampleObj['playbackRate']
        resampleFilter  = sampleObj['resampleFilter']
        ampBoost        = sampleObj['ampBoost']

        sampleData  = sampleData.astype(numpy.float64)
        sampleData *= ampBoost
        sampleData  = numpy.clip(sampleData, numpy.iinfo(numpy.int16).min, numpy.iinfo(numpy.int16).max).astype(numpy.int16)

        # use resampy to resample down to target frequency
        if sr_orig != playbackRate:
            newSampleData = sampleData.astype(numpy.float32)
            newSampleData = resampy.resample(sampleData, sr_orig, playbackRate, filter=resampleFilter)
            newSampleData = numpy.clip(newSampleData, numpy.iinfo(numpy.int16).min, numpy.iinfo(numpy.int16).max).astype(numpy.int16)
        else:
            newSampleData = sampleData

        newSampleData = newSampleData.astype(numpy.int32)
        newSampleData += 32767
        newSampleData = numpy.clip(newSampleData, 0, 65535)

        eightbitSampleData = newSampleData >> 8
        fivebitSampleData  = newSampleData >> 11
        fourbitSampleData  = newSampleData >> 12

        eightbitSampleData = numpy.clip(eightbitSampleData, 0, 255)
        fivebitSampleData  = numpy.clip(fivebitSampleData, 0, 31)

        return '', fivebitSampleData.tolist(), eightbitSampleData.tolist(), fourbitSampleData.tolist()


class WavRead():

    def __init__(self, filename, debug=False):
        self.wavHeader      = {}
        self.wavContents    = []
        self.contentIndex   = 0
        self.content        = None
        self.filename       = filename
        self.message        = ''
        self.debug          = debug

    def readFile(self):
        try:
            with open(self.filename,'rb') as f:
                self.content = f.read()
                self.content = [int(item) & 0xff for item in self.content]
        except Exception as e:
            print(f'Error reading contents: {e}')
            return False

        print('File read.')

        result = self.content != None and self.getWavHeader() and self.getWaveData()

        return result, self.message, self.wavHeader, self.wavContents


    # ///////////////////////////////////////////////////////////////////////
    def getWavHeader(self):

        RiffID          = self.read4cc()
        RiffIDsize      = self.intReadLE()
        WaveID          = self.read4cc()

        FmtID           = self.read4cc()
        FMTsize         = self.intReadLE()
        fmtType         = self.wordReadLE()

        Channels        = self.wordReadLE()

        SamplesPerSec   = self.intReadLE()
        AvgBytesPerSec  = self.intReadLE()

        BlockAlign      = self.wordReadLE()
        BitsPerSample   = self.wordReadLE()

        DataID          = self.read4cc()
        DataIDsize      = self.intReadLE()

        if RiffID != 'RIFF':
            self.message = "\n Possible incorrect or corrupt wave file. No RIFF ID. \n"
        elif WaveID != 'WAVE':
            self.message = "\n Possible incorrect or corrupt wave file. No WAVE ID. \n"
        elif FmtID != 'fmt ':
            self.message = "\n Possible incorrect or corrupt wave file. No FMT ID. \n"
        elif DataID != 'data':
            self.message = "\n Possible incorrect or corrupt wave file. No DATA ID. \n"
        elif FMTsize != 0x10:
            self.message = "\n Unsupported RIFF FMT format or correct wavefile. \n"
        elif (BitsPerSample!=8) and (BitsPerSample!=16):
            self.message = f"\n Wave file must be 8bit or 16bit in depth. Found: {BitsPerSample}bit\n"
        elif Channels > 2:
            self.message = f"\n More than 2 channel support isn't currently implemented. \n"
        else:
            self.wavHeader = {
                'RiffID'            : RiffID,
                'RiffIDsize'        : RiffIDsize,
                'WaveID'            : WaveID,
                'FMT'               : FmtID,
                'FMTsize'           : FMTsize,
                'fmtType'           : fmtType,
                'Channels'          : Channels,
                'SamplesPerSec'     : SamplesPerSec,
                'AvgBytesPerSec'    : AvgBytesPerSec,
                'BlockAlign'        : BlockAlign,
                'BitsPerSample'     : BitsPerSample,
                'DataID'            : DataID,
                'DataIDsize'        : DataIDsize
            }

            self.message  = f" Bit depth: {BitsPerSample}\n"
            self.message += f" Channels:  {Channels}\n"
            self.message += f" Rate:      {SamplesPerSec}\n"

        print(self.message)
        print(self.wavHeader)

        return self.wavHeader != {}

    # ///////////////////////////////////////////////////////////////////////
    def getWaveData(self):

        numChans    = self.wavHeader['Channels']
        pcmSize     = self.wavHeader['BitsPerSample'] // 8
        step        = pcmSize
        DataIDsize = self.wavHeader['DataIDsize']

        if self.debug:
            print(f' {self.contentIndex}, {len(self.content)}, {step}, {pcmSize}, {DataIDsize}')

        sampleType = (numpy.int16,numpy.uint8)[pcmSize==1]
        wavContents = numpy.fromfile(file=self.filename,dtype=sampleType,offset=self.contentIndex, count=DataIDsize // pcmSize)

        if pcmSize==1:
            wavContents = wavContents.astype(numpy.int16)
            wavContents -= 128
            wavContents *= 256

        wavContents = numpy.clip(wavContents, numpy.iinfo(numpy.int16).min, numpy.iinfo(numpy.int16).max).astype(numpy.int16)

        # convert to mono
        if numChans != 1:
            wavContents = wavContents.astype(numpy.int32)
            wavContents = wavContents.reshape(-1, 2)
            wavContents = ((wavContents[:,0] + wavContents[:,1]) / 2)
            wavContents = numpy.clip(wavContents, numpy.iinfo(numpy.int16).min, numpy.iinfo(numpy.int16).max).astype(numpy.int16)

        self.wavContents = wavContents

        return self.wavContents.size != 0

    # ///////////////////////////////////////////////////////////////////////
    # Helper functions
    # ///////////////////////////////////////////////////////////////////////

    def read4cc(self):
        data = self.content[self.contentIndex : self.contentIndex+4 ]
        self.contentIndex += 4
        return ''.join([chr(val) for val in data])

    def intReadBE(self):
        idx = self.contentIndex
        self.contentIndex += 4
        val  = self.content[idx+0:idx+1][0] << 24
        val |= self.content[idx+1:idx+2][0] << 16
        val |= self.content[idx+2:idx+3][0] <<  8
        val |= self.content[idx+3:idx+4][0] <<  0
        return val

    def intReadLE(self):
        idx = self.contentIndex
        self.contentIndex += 4
        val  = self.content[idx+0:idx+1][0] <<  0
        val |= self.content[idx+1:idx+2][0] <<  8
        val |= self.content[idx+2:idx+3][0] << 16
        val |= self.content[idx+3:idx+4][0] << 24
        return val

    def wordReadBE(self):
        idx = self.contentIndex
        self.contentIndex += 2
        val  = self.content[idx+0:idx+1][0] << 8
        val |= self.content[idx+1:idx+2][0] << 0
        return val

    def wordReadLE(self):
        idx = self.contentIndex
        self.contentIndex += 2
        val  = self.content[idx+0:idx+1][0] << 0
        val |= self.content[idx+1:idx+2][0] << 8
        return val

class GuiFrontend():

    def __init__(self, args):
        self.pcmData    = []
        self.pcmHeader  = {}
        self.components = {}
        self.args       = args
        self.filename    = ''
        self.orgWavefile = ''
        self.DDAdate     = []

    def plotWave(self, *args):

        waveType = args[0]
        self.amplified = int(self.components['ampBoost'].get()[0:-1])/100
        adjust_vol = 1
        if self.components['plotWithVol'].get() == 1:
            adjust_vol = self.getLinearVolume()

        self.root.update()

        if waveType == '4bit':
            y = numpy.asarray(self.DDAdata[2],dtype=numpy.uint8)*16
            x = [i for i in range(len(self.DDAdata[0]))]
        elif waveType == '5bit':
            y = numpy.asarray(self.DDAdata[0],dtype=numpy.uint8)*8
            x = [i for i in range(len(self.DDAdata[0]))]
        elif waveType == '8bit':
            y = numpy.asarray(self.DDAdata[1],dtype=numpy.uint8)
            x = [i for i in range(len(self.DDAdata[1]))]

        ceiling = numpy.asarray([255.0 for _ in range(len(self.DDAdata[1]))],dtype=numpy.uint8)
        floor   = numpy.asarray([0.0 for _ in range(len(self.DDAdata[1]))],dtype=numpy.uint8)
        y = y*adjust_vol
        y = y + ((255-(255*adjust_vol))/2)

        rms = numpy.sqrt(numpy.mean(numpy.square(y)))
        print("RMS:", rms)

        plt.plot(x, ceiling)
        plt.plot(x, floor)
        plt.plot(x, y)
        plt.xlabel(f'x (amp={self.amplified}x)')
        plt.ylabel(f'Amplitude (max range: 0 - 255)')
        plt.title('Converted Wave')
        plt.show(block=False)


    def openWave(self):

        try:
            self.play_obj.stop()
        except:
            pass

        filename = fd.askopenfilename(defaultextension='.wav', filetypes = (("wav files","*.wav"),("all files","*")))
        if filename == '' or filename == None:
            print('Cancel open..')
            return

        result, convertInfo, self.pcmHeader, self.pcmData = WavRead(filename).readFile()

        if not result and tk.messagebox.showerror(title="Failed...", message=convertInfo):
            return

        tk.messagebox.showinfo(title="Wav/RIFF Info", message=convertInfo)
        self.componentState(tk.NORMAL)
        self.filename = self.sfxnameVar.set(pathlib.Path(filename).stem)
        self.orgWavefile = filename

        playbackRate = self.components['playback'].get()
        self.pcmHeader['resampleFilter'] = self.components['filter'].get()
        self.pcmHeader['playbackRate']   = int( (playbackRate,self.pcmHeader['SamplesPerSec'])[playbackRate == "no resample"] )
        self.pcmHeader['ampBoost']       = int(self.components['ampBoost'].get()[0:-1])/100

        message, newSampleData, eightBitData, fourBitData = ConvertWave().convertPCMData(self.pcmHeader,self.pcmData)

        self.DDAdata = [newSampleData, eightBitData, fourBitData]

        return result

    def convertWavefile(self):
        playbackRate = self.components['playback'].get()
        self.pcmHeader['resampleFilter'] = self.components['filter'].get()
        self.pcmHeader['playbackRate']   = int( (playbackRate,self.pcmHeader['SamplesPerSec'])[playbackRate == "no resample"] )
        self.pcmHeader['ampBoost']       = int(self.components['ampBoost'].get()[0:-1])/100

        message, newSampleData, eightBitData, fourBitData = ConvertWave().convertPCMData(self.pcmHeader,self.pcmData)

        self.DDAdata = [newSampleData, eightBitData, fourBitData]

    def saveFile(self):

        saveDir = fd.askdirectory()
        playbackRate = self.components['playback'].get()
        self.pcmHeader['resampleFilter'] = self.components['filter'].get()
        self.pcmHeader['playbackRate']   = int( (playbackRate,self.pcmHeader['SamplesPerSec'])[playbackRate == "no resample"] )
        self.pcmHeader['ampBoost']       = int(self.components['ampBoost'].get()[0:-1])/100

        message, newSampleData, eightBitData, fourBitData = ConvertWave().convertPCMData(self.pcmHeader,self.pcmData)

        self.DDAdata = [newSampleData, eightBitData, fourBitData]

        # TODO, need to add a switch in the GUI for this
        newSampleData = [(item,0x01)[item==0x00] for item in eightBitData]
        newSampleData = newSampleData + [0x00]

        tk.messagebox.showinfo(title=None, message='HuPCM file saved.')

        filename = (self.sfxnameVar.get().strip(),self.filename)[self.sfxnameVar.get().strip() == ""]
        includePath = self.includePath.get().strip()

        with open(f'{os.path.join(saveDir,filename)}.inc','w') as f:

            # TODO needs to be a GUI option
            f.write(f'\n')
            f.write(f'  .db bank(.sample)\n')
            f.write(f'  .dw .sample\n\n')
            f.write(f'.sample\n\n')
            f.write (f'  .page {2}\n\n')
            f.write(f'  .include \"{os.path.join(includePath,filename)}.data.inc\"\n\n')

        with open(f'{os.path.join(saveDir,filename)}.data.inc','w') as f:
            columnBytes = 0
            sampleCount = 0
            for idx, val in enumerate(newSampleData):
                valString = hex(val)[2:]
                valString = '$'+('','0')[len(valString) == 1] + valString
                if columnBytes == 0:
                    f.write("  .db ")
                f.write(f'{valString}')
                columnBytes += 1
                if columnBytes > 15:
                    f.write("\n")
                    columnBytes = 0
                    if sampleCount >= 16384:
                        f.write(f'\n\n  .page {2}\n\n')
                        sampleCount = 0
                elif idx == len(newSampleData)-1:
                    f.write("\n")
                else:
                    f.write(", ")
                sampleCount += 1

        with open(f'{filename}.debug.8bit.bin','wb') as f:
            f.write(bytearray(newSampleData))

        comp_4bit_output = []
        last_sample = -1
        for grp in range(0,(len(fourBitData)//8)*8,8):
            mask = 0
            count = 0
            temp = []
            for i in range(grp,grp+8,1):

                curr_sample = fourBitData[i]
                if curr_sample == last_sample:
                    mask |= (1<<count)
                else:
                    temp.append(curr_sample)

                last_sample = curr_sample
                count += 1
            comp_4bit_output = comp_4bit_output + [mask] + temp[:]

        print(f' org size: {len(fourBitData)}. New size: {len(comp_4bit_output)}')
        with open(f'{filename}.4bit.bin','wb') as f:
            f.write(bytearray(comp_4bit_output))
        with open(f'{filename}.raw.4bit.bin','wb') as f:
            f.write(bytearray(fourBitData))


    def componentState(self, compState):
            for child in self.components["subframe1"].winfo_children():
                child.configure(state=compState)
            for child in self.components["subframe2"].winfo_children():
                child.configure(state=compState)
            self.components["save"].config(state=compState)


    def playOriginal(self):
        if not self.orgWavefile and tk.messagebox.showerror(title="No wavefile", message='Please import a wavefile before trying to play.'):
            return
        self.wave_obj = sa.WaveObject.from_wave_file(self.orgWavefile)
        self.play_obj = self.wave_obj.play()

    def stopPlayback(self):
        self.play_obj.stop()


    def playConvert(self):
        pass

    def getLinearVolume(self):

        chan_vol_dB = self.volList.index(self.components['pceVol'].get()) * -1.5
        lin_vol = 10**(chan_vol_dB/20)
        print(f"Lin vol: {lin_vol}")
        return lin_vol

    def playPCE_4bit(self):
        try:
            self.play_obj.stop()
        except:
            pass
        vol_adjust = self.getLinearVolume()
        waveData = convertFreq(numpy.asarray(self.DDAdata[2],dtype=numpy.uint8),44100,self.pcmHeader['playbackRate'],expand4bit=True,vol_adjust=vol_adjust)
        print(f" org: {len(self.DDAdata[0])}, {len(waveData)} , {len(waveData)/len(self.DDAdata[0])}, {44100/self.pcmHeader['playbackRate']}")
        audio_data = bytearray(waveData)
        num_channels = 1
        bytes_per_sample = 1
        sample_rate = 44100
        self.play_obj = sa.play_buffer(audio_data, num_channels, bytes_per_sample, sample_rate)

    def playPCE_5bit(self):
        try:
            self.play_obj.stop()
        except:
            pass
        vol_adjust = self.getLinearVolume()
        waveData = convertFreq(numpy.asarray(self.DDAdata[0],dtype=numpy.uint8),44100,self.pcmHeader['playbackRate'],expand5bit=True,vol_adjust=vol_adjust)
        print(f" org: {len(self.DDAdata[0])}, {len(waveData)} , {len(waveData)/len(self.DDAdata[0])}, {44100/self.pcmHeader['playbackRate']}")
        audio_data = bytearray(waveData)
        num_channels = 1
        bytes_per_sample = 1
        sample_rate = 44100

        self.play_obj = sa.play_buffer(audio_data, num_channels, bytes_per_sample, sample_rate)

    def playPCE_8bit(self):
        try:
            self.play_obj.stop()
        except:
            pass
        vol_adjust = self.getLinearVolume()
        waveData = convertFreq(numpy.asarray(self.DDAdata[1],dtype=numpy.uint8),44100,self.pcmHeader['playbackRate'],expand5bit=False,vol_adjust=vol_adjust)
        print(f" org: {len(self.DDAdata[0])}, {len(waveData)} , {len(waveData)/len(self.DDAdata[0])}, {44100/self.pcmHeader['playbackRate']}")
        audio_data = bytearray(waveData)
        num_channels = 1
        bytes_per_sample = 1
        sample_rate = 44100

        self.play_obj = sa.play_buffer(audio_data, num_channels, bytes_per_sample, sample_rate)

    def process(self):
        root = tk.Tk()
        self.components['root'] = root
        root.title("Wav/RIFF to HuPCM Converter")


        frame1 = tk.LabelFrame(root,padx=24, pady=24)
        frame1.pack()

        subframe1 = tk.LabelFrame(frame1, padx=4, pady=4)
        subframe1.pack(side=tk.LEFT)
        self.components['subframe1'] = subframe1

        labelTop = ttk.Label(subframe1, text = "Filter Type")
        labelTop.grid(column=0, row=0)
        filterCombo = ttk.Combobox(subframe1, values=filterValues)
        self.components['filter'] = filterCombo
        filterCombo.grid(column=0, row=1)
        filterCombo.current(0)

        labelTop = ttk.Label(subframe1, text = "Bit Depth")
        labelTop.grid(column=0, row=2)
        bitdepthCombo = ttk.Combobox(subframe1, values=bitdepthValues)
        self.components['bitdepth'] = bitdepthCombo
        bitdepthCombo.grid(column=0, row=3)
        bitdepthCombo.current(0)

        labelTop = ttk.Label(subframe1, text = "Playback Rate")
        labelTop.grid(column=0, row=4)
        playerateCombo = ttk.Combobox(subframe1, values=playerateValues)
        self.components['playback'] = playerateCombo
        playerateCombo.grid(column=0, row=5)
        playerateCombo.current(0)

        labelTop = ttk.Label(subframe1, text = "Debug")
        labelTop.grid(column=0, row=6)
        debugCombo = ttk.Combobox(subframe1, values=['Off','On'])
        self.components['debug'] = debugCombo
        debugCombo.grid(column=0, row=7)
        debugCombo.current(0)

        labelTop = ttk.Label(subframe1, text = "Amplitude boost")
        labelTop.grid(column=0, row=8)
        selected_value = tk.StringVar()
        ampBoostCombo = ttk.Combobox(subframe1, textvariable=selected_value)
        ampBoostCombo['values']=['100%','105%','110%','115%','125%','137%','150%','165%','175%','185%','195%','210%','220%','230%','240%','250%','275%','300%','325%','350%','375%','400%','425%','450%']
        ampBoostCombo.set('137%')
        self.components['ampBoost'] = selected_value
        ampBoostCombo.grid(column=0, row=9)


        subframe2 = tk.LabelFrame(frame1, padx=4, pady=4)
        subframe2.pack(side=tk.LEFT)
        self.components['subframe2'] = subframe2

        labelTop = ttk.Label(subframe2, text = "Include Path: ")
        labelTop.grid(column=0, row=0)
        self.includePath = tk.StringVar()
        includePath = tk.Entry(subframe2,textvariable=self.includePath)
        self.components['path'] = includePath
        includePath.grid(column=1, row=0)
        labelTop = ttk.Label(subframe2, text = "PCM name: ")
        labelTop.grid(column=0, row=1)
        self.sfxnameVar = tk.StringVar()
        sfxname = tk.Entry(subframe2,textvariable=self.sfxnameVar )
        self.components['sfxname'] = sfxname
        sfxname.grid(column=1, row=1)




        subframe3 = tk.LabelFrame(frame1, padx=4, pady=4)
        subframe3.pack(side=tk.LEFT)
        self.components['subframe3'] = subframe3

        openButton   = tk.Button(subframe3, text='     Open WAV       ', command=self.openWave)
        openButton.grid(row=0, column=0,sticky=tk.W)

        convWaveButton   = tk.Button(subframe3, text='     Convert HuPCM    ', command=self.convertWavefile)
        convWaveButton.grid(row=2, column=0,sticky=tk.W)

        saveButton   = tk.Button(subframe3, text='     Save HuPCM    ', command=self.saveFile)
        saveButton.grid(row=2, column=1,sticky=tk.W)
        self.components["save"] = saveButton
        playOrgButton   = tk.Button(subframe3, text='   Play Original   ', command=self.playOriginal)
        playOrgButton.grid(row=3, column=0,sticky=tk.W)
        playConvertButton   = tk.Button(subframe3, text='   Play Converted   ', command=self.playConvert)
        playConvertButton.grid(row=4, column=0,sticky=tk.W)

        playDDA4bitButton   = tk.Button(subframe3, text='   Play PCE 4bit', command=self.playPCE_4bit)
        playDDA4bitButton.grid(row=5, column=0,sticky=tk.W)
        playDDA5bitButton   = tk.Button(subframe3, text='   Play PCE 5bit', command=self.playPCE_5bit)
        playDDA5bitButton.grid(row=5, column=1,sticky=tk.W)
        playDDA8bitButton   = tk.Button(subframe3, text='   Play PCE 8bit', command=self.playPCE_8bit)
        playDDA8bitButton.grid(row=5, column=2,sticky=tk.W)


        stopPlaybackButton   = tk.Button(subframe3, text='   stop playback   ', command=self.stopPlayback)
        stopPlaybackButton.grid(row=6, column=0,sticky=tk.W)

        withVol = tk.IntVar()
        withVol.set(1)
        self.components['plotWithVol'] = withVol
        withVolBox = tk.Checkbutton(subframe3,text='Show Plot w/Vol',variable=withVol, onvalue=1)
        withVolBox.grid(row=7, column=0,sticky=tk.W)

        plot4bitButton   = tk.Button(subframe3, text='   plot 4bit   ', command=partial(self.plotWave,'4bit'))
        plot4bitButton.grid(row=8, column=0,sticky=tk.W)
        plot5bitButton   = tk.Button(subframe3, text='   plot 5bit   ', command=partial(self.plotWave,'5bit'))
        plot5bitButton.grid(row=8, column=1,sticky=tk.W)
        plot8bitButton   = tk.Button(subframe3, text='   plot 8bit   ', command=partial(self.plotWave,'8bit'))
        plot8bitButton.grid(row=8, column=2,sticky=tk.W)

        labelTop = ttk.Label(subframe3, text = "PCE Channel Volume")
        labelTop.grid(column=0, row=9)
        selected_vol = tk.StringVar()
        self.volList = [ '31','30','29','28','27','26',
                         '25','24','23','22','21','20',
                         '19','18','17','16','15','14',
                         '13','12','11','10','9','8',
                         '7','6','5','4','3','2','1' ]
        pceChanCombo = ttk.Combobox(subframe3, textvariable=selected_vol)
        pceChanCombo['values'] = self.volList[:]
        pceChanCombo.set('31')
        self.components['pceVol'] = selected_vol
        pceChanCombo.grid(column=0, row=10)


        self.componentState(tk.DISABLED)

        self.root = root
        root.mainloop()


#############################################################################################################
#############################################################################################################
#............................................................................................................
#                                                                                                           .
# Main                                                                                                      .
#............................................................................................................

def auto_int(val):
    val = int(val, (DECbase,HEXbase)['0x' in val])
    return val

parser = argparse.ArgumentParser(description='Convert WAV to PCE PCM.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

runOptionsGroup = parser.add_argument_group('Run options', 'Run options for DMF converter')
runOptionsGroup.add_argument('--destinationPath',
                                '-destpth',
                                required=False,
                                default="",
                                help='Copies the newly created files to a specific path.')
runOptionsGroup.add_argument('--includePath',
                                '-incpth',
                                required=False,
                                default="",
                                help='Relative path prefix for file ".include"')
runOptionsGroup.add_argument('--NoSongNameSubfolder',
                                '-nosub',
                                required=False,
                                action="store_true",
                                help='Stops util from using the song name as a sub folder.')
runOptionsGroup.add_argument('--bitpackPCM',
                                '-pack',
                                action="store_true",
                                default=False,
                                help='Bit packs the 5bit the samples.')
runOptionsGroup.add_argument('--debug',
                                '-dbg',
                                default=False,
                                action="store_true",
                                help='Output uncompressed DMF as raw bin and hex s-record.')
runOptionsGroup.add_argument('--alignPCM256',
                                '-align256',
                                action="store_true",
                                default=False,
                                help='Forces all samples to take up a multiple of 256 bytes and block aligns to 256byte boundaries - remaining values will be 0.')
runOptionsGroup.add_argument('--resampleFilter',
                                '-refil',
                                choices=['kaiser_best','kaiser_fast','sinc_window_32','sinc_window_Hann','default'],
                                default='kaiser_best',
                                help='See https://resampy.readthedocs.io/ documentation for info on filters.')
runOptionsGroup.add_argument('--bitDepth',
                                '-bdpth',
                                choices=['8bit'],
                                default='8bit',
                                help='The bit depth for streaming PCM samples.')
runOptionsGroup.add_argument('--playback',
                                '-pb',
                                choices=['6960','6991','7020','9279','10440','11090','12180','13920','14040'],
                                default='6960',
                                help='The playback rate for PCM samples.')
runOptionsGroup.add_argument('--noGui',
                                '-ng',
                                action="store_true",
                                default=False,
                                help='The bit depth for streaming PCM samples.')

args = parser.parse_args()

runOptions['destinationPath'] = args.destinationPath
runOptions['subFolder']       = ('songName','')[args.NoSongNameSubfolder]
runOptions['includePath']     = args.includePath
runOptions['bitpackPCM']      = args.bitpackPCM
runOptions['bitDepth']        = args.bitDepth
runOptions['resampleFilter']  = args.resampleFilter
runOptions['playback']        = args.playback
runOptions['alignPCM256']     = args.alignPCM256
runOptions['debug']           = args.debug

if not args.noGui:
    GuiFrontend(args).process()
else:
    print(f'Only GUI mode is operational')
