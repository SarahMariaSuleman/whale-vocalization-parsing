"""
Course: CSC 475 - Music Retreival Techniques
Group Project - Orca
Last Modified: April 7th, 2024
"""

#Imports
import scipy.io.wavfile as wav
import librosa as lb
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import datetime as dt
from sklearn.linear_model import LinearRegression

#Switches and Settings for looping
call_type="C10"
pod_name="N16"
parse_all_calls=True
temp=1

#Initialize Pandas DataFrame with column names
calls=pd.DataFrame(columns=["pod name","call type","date","random info","call id","FF average","datetime object","date ordinal","year"])

#DATA COLLECTION AND FF ESTIMATION

#Parse the Mini Orchive WAV file folder
for subdir, dirs, files in os.walk("mini_orchive/call-catalog/wav"):
    
    #For all files
    for file in files:

        #If the call type and pod name specified are found in the WAV file name
        #if ((call_type in file)):
        if(temp==1):
                
                #Split the file name by - to extract the call information
                f_info = file.split("-")

                #Extract the call information
                if os.path.exists(subdir + "/" +file):
                    
                    #Parse file to check for correct formatting RIFF
                    file_path = subdir+"/" + file
                    tester = open(file_path, "rb")
                    header = tester.read(44)

                    #If formatting is readable
                    if (header[:4] == b'RIFF'):

                        #Open WAV file
                        srate, audio_signal = wav.read(file_path)

                        #Convert WAV file to float values
                        audio_signal2=[]
                        for value in audio_signal:
                            audio_signal2.append(float(value))
                
                        #Convert to Numpy array
                        audio_signal3  = np.array(audio_signal2)

                        #Compute the Fundamental Frequency estimation
                        FF_est=lb.yin(audio_signal3, fmin=200, fmax= 2000, sr=srate)

                        #Compute the mean of the FF estimation
                        FF_mean = np.mean(lb.yin(audio_signal3, fmin=200, fmax= 2000, sr=srate))

                        #Convert the date info in file name to datetime object
                        date1 = dt.datetime.strptime(f_info[2],'%m%d%y').date()

                        #Get date ordinal number from datetime object(If computing linear regression with individual dates and not year grouping)
                        date_ordinal= dt.datetime.toordinal(date1)

                        #Create a new dataframe row
                        new_row = {'pod name': f_info[0], 'call type':f_info[1], 'date':f_info[2], 'random info':f_info[3], 'call id':f_info[4], "FF average":FF_mean, "datetime object": date1, "date ordinal": date_ordinal, "year": date1.year}
                        
                        #Add row to calls DataFrame
                        calls.loc[len(calls)] = new_row
                      

#LINEAR REGRESSION CALCULATION

#Create a new dataframe only consisting of the columns FF average and year
new=calls[["FF average", "year"]].copy()

#Group values in the FF average column by year value and then calculate average
by_year = new.groupby("year", as_index=False)["FF average"].mean()

#Create a scatter plot 
plt.scatter(by_year["year"], by_year["FF average"])

#Perform linear regression using sklearn library
lin_reg = LinearRegression()

#Fit the model to the data
lin_reg.fit(by_year["year"].to_numpy().reshape(-1,1), by_year["FF average"].to_numpy())

#Create line based on intercept and coefficent returned from linear regression model
y_vals = lin_reg.intercept_ + lin_reg.coef_ * by_year["year"].to_numpy().reshape(-1,1)

#Plot line to show trend
plt.plot(by_year["year"].to_numpy().reshape(-1,1), y_vals, 'r-')

#Show plot
plt.show()


