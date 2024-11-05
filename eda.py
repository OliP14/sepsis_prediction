import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


DATA_A_PATH = "/Users/oliverpasquesi/School/CPSC4300/sepsis_prediction/training/training_setA/"


# Open 1st file, get header names. Make df for dataset_A, populate with header, add cols, populate with contents of file 1
# Open any file, get headers, set headers for dataset_A
temp = pd.read_csv('/Users/oliverpasquesi/School/CPSC4300/sepsis_prediction/training/training_setA/p018305.psv', sep='|')
header = temp.columns
dataset_A = pd.DataFrame(columns=header)
dataset_A.insert(0, 'patient_id', [])
dataset_A.insert(0, 'hospital', [])
#print(dataset_A.columns)

# Make a list to store all individual patient dfs
patient_dfs = []

# Construct strings to iterate open all files: 'p' + int_str.zfill(6) + '.psv'<-- zfill adds leading 0s
# Iterate through all files, open, construct temp df: populate with additional data, add to full dataset
# Should I convert the hour to 24 hr format? We are given the admittance time so we would be able to derive it
for patient_id in range(1, 20644): # set upper limit to 20644 for prod
    filename = 'p' + str(patient_id).zfill(6) + '.psv'
    print(f"opening: {filename}")
    try:
        file_df = pd.read_csv(DATA_A_PATH + filename, sep='|')
        file_df.insert(0, 'patient_id', patient_id)
        file_df.insert(0, 'hospital', 'A')
        patient_dfs.append(file_df)
    except OSError as e:
        print(e)
        pass

# Join all of the patient dfs into a single large df
dataset_A = pd.concat(patient_dfs)
print(dataset_A.shape)
print(np.sum(dataset_A.isnull()))
print(dataset_A.info())

# clean data: remove all nan cols, remove rows with nan hr?

# How to make multiple plots and specify what data items to put on each? I want a plot for sepsis and a plot for no sepsis
#   Do I need to use groupby()?

"""
# parse data, plot heart rate throughout duration of stay. Make 2 plots, those w sepsis, those w/out
sep_plt = plt.figure()
plt.title("sepsis hr plot")
no_sep_plt = plt.figure()
plt.title("no sepsis hr plot")
sep_count = 0
no_sep_count = 0
for patient in range(0,2043):
    hour = patient_dfs[patient]['ICULOS'] # x
    hr = patient_dfs[patient]['HR'] # y
    sep = patient_dfs[patient]['SepsisLabel']
    #print(f"sep value is: {sep[0]}")
    if sep[0] == 0:
        no_sep_count += 1
        plt.plot(hour, hr)
    else:
        sep_count += 1
        plt.plot(hour, hr)
print(f"Sep count: {sep_count}")
print(f"no sep count: {no_sep_count}")
plt.show()
"""

# Comparing vars:
#   x: time
#   y: hr, etc.
# plot 2 separate lines: those who had sepsis and those who didn't. Compare how the lines are different to see relationship between changing vitals and if it is indicative of sepsis
# scatterplot all vars (individually/on individual plot), make sepsis red dot, no sepsis blue dot. Can we use clusters to identify sepsis?
#   take just a single datapoint from each patient file (take 2nd entry (not 1st bc its missing some vals))
#       > The importance of this representation hinges on the fact if sepsis is developed or if someone has it before admittance to the hospital
#
# try and represent the data as line graphs (over time of stay). split data by who has and who doesn't have sepsis. stack the line graphs over each other, see if
# trends persist. Where you find a trend is a meaningful indicator of sepsis
#   > do lin reg on each individual data point vs time. Also do a poly reg on all data points combined
# 
# theres a lot of nan vals... what to do w them? If we delete all nan then it would remove a lot of data. Some things just aren't measured as regularly as others
# so nan vals should just be empty. The data point would just be plotted as having a higher time frame.
#   if all nan, remove the column.

# How should I store the data? It might not make sense to combine everything into a single df and it might be easier to have a list/array of dfs. This would
# better separate the data. When doing stats, plotting etc., I can just apply the same action/operation on every patient_df in the list
#   But then, if we combine all into a single df, adding additional labels would let us be able to differentiate between patients


"""
find what features have the most null values, remove columns with majority null, remove entries (rows) with nulls?
    find what features (columns) have least (ideally no) nan values


pseudo:
make df of hospital A, make header from header of 1st file (exclude header (row 1) when reading rest of files)
    add cols: hospital (A/B), patient_id (name of file)
for patient_id in range(0, __):
    read file patient_id.psv
    add values to cols hospital and patient_id
    append patient df to hospital df


what is each psv file represent? Is it a single patient?
    what does each entry in individual file represent? is it time-series data?
    |
    |
    +-> Each file represents a single patient (idk about repeat customers)
        Each entry represents the hourly data recorded (for each hour of the patient's stay)
            8 vital, 26 lab, 6 demographic variables

        Each dataset (A, B) data collected from a different hospital (ICU)

consider/learn:
- what is sepsis?
- what hypothesis can we make about predicting sepsis?
    - how do we arrive at that conclusion? have exploratory data analysis + interpretation to back that up
"""
