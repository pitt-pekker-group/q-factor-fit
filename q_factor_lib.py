import numpy as np
from nptdms import TdmsFile
import pandas as pd

from matplotlib import pyplot
from scipy.optimize import minimize

#from scipy.signal import butter, filtfilt

def model_function(fix_arg, params):
    # Generate the S21 curve based on the model parameters.
    
    # params is a list or array of parameters for the model
    dw,w0 = fix_arg #angular frequency [2pi*GHz]
    R0,L0,a,b,Z1_r,Z1_i = params #fitting parameters
    
    Z1 = Z1_r + 1j * Z1_i
    gamma = a+b*1j
    
    Z = (R0+2*1j*dw*L0) #impedence of the circuit around the resonate frequency
    Z0 = 50.0 #characteristic impedence of transmission line
    
    return (2*Z *Z0 *Z1)/((Z0*np.cosh(gamma) + Z1*np.sinh(gamma))*((2*Z + Z0)*Z1*np.cosh(gamma) + (2*Z*Z0 + Z1*np.conjugate(Z1))*np.sinh(gamma)))



def loss_function(params, fix_arg, S21):
    #dw,w0 = fix_arg
    predictions = model_function(fix_arg, params)
    return np.sum(abs(predictions - S21))  


def get_col(df,par):
    col=df.columns
    col=[par in c for c in col]
    val=df.iloc[:,col].values
    if val.shape[1]==1:
        return val[:,0]
    return np.nan

def get_val(df, par):
    c=get_col(df,par)
    if isinstance(c,np.ndarray):
        return c[0]
    return(np.nan)

def read_data(fname):
    tdms_file = TdmsFile.read(fname)
    df=tdms_file.as_dataframe()
    col=df.columns

    print(col)
    
    # Figure out the experimental cinditions
    B=get_val(df,'Magnet Field')
    if np.isnan(B):
        B=get_val(df,'Magnet')
    T=get_val(df,'Temperature')
    
    # Figure out the transmission measurements
    freq=get_col(df,'freq')
    ReS21=get_col(df,'ReS21')
    ImS21=get_col(df,'ImS21')
    S21=ReS21+1j*ImS21

    df_out=pd.DataFrame({'freq': freq, 'ReS21': ReS21, 'ImS21': ImS21, 'S21':S21})

    return(B,T,df_out)

def plot_data(df, col, ax, select=[0,-1], plot_axis=False):
    ax.set_xlabel('f [GHz]')
    ax.set_ylabel('Transmission matrix element')
    
    ax.plot(df['freq'][select[0]:select[1]]/1E9,np.real(df[col][select[0]:select[1]]),'--',label = 'Re$(S_{21})$')
    ax.plot(df['freq'][select[0]:select[1]]/1E9,np.imag(df[col][select[0]:select[1]]),'--',label = 'Im$(S_{21})$')
    ax.plot(df['freq'][select[0]:select[1]]/1E9,np.abs(df[col][select[0]:select[1]]),'--',label = 'Abs$(S_{21})$')

    if plot_axis:
        ax.plot([df['freq'][select[0]]/1E9,df['freq'][select[1]]/1E9],[0,0],'-',color='black', linewidth=0.5)
    
    ax.minorticks_on()
    ax.legend()

def remove_fast_osc(df):
    freq_step = df['freq'][1] - df['freq'][0]                      
    fft_freq = np.fft.fftfreq(len(df), d=freq_step)        
    fft_data = np.fft.fft(df['S21'])
    
    pos_max=np.argmax(np.abs(fft_data))
    freq_max=fft_freq[pos_max]
    print('Fast Oscillation Freq: ',freq_max)
    
    pyplot.plot(fft_freq[pos_max-100:pos_max+100]*1E9,np.abs(fft_data)[pos_max-100:pos_max+100],'.')
    pyplot.plot(freq_max*np.array([1,1])*1E9,[0,1.05*np.abs(fft_data[pos_max])],'--')
    pyplot.xlabel('1/freq [1/GHz]')
    pyplot.ylabel('FFT Abs[S21]')
    pyplot.show()
    
    df['S21_filtered']=df['S21']*np.exp(-1j*2*np.pi*df['freq']*freq_max)

def plot_circles(df, col, ax, s1, s2):
    ax.plot(np.real(df[col]),np.imag(df[col]))
    ax.plot(np.real(df[col][s1:s2]),np.imag(df[col][s1:s2]))
    
    ax.plot([min(np.real(df[col])),max(np.real(df[col]))],[0,0],'-',color='black',linewidth=0.5)
    ax.plot([0,0],[min(np.imag(df[col])),max(np.imag(df[col]))],'-',color='black',linewidth=0.5)
    
    ax.set_xlim(min(np.real(df[col])),max(np.real(df[col])))
    ax.set_ylim(min(np.imag(df[col])),max(np.imag(df[col])))
        
    ax.set_xlabel('Re [S21]')
    ax.set_ylabel('Im [S21]')