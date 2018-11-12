import pandas as pd
import numpy as np
# for Box-Cox Transformation
from scipy import stats
# for min_max scaling

# plotting modules
import seaborn as sns
import matplotlib.pyplot as plt

import chardet
import fuzzywuzzy
from fuzzywuzzy import process

inputData = pd.read_csv("C:\\Users\\pt2\\PycharmProjects\\Kaggle_ML_Cource\\data_cleaning_challenge\\All_Opp_27032018.csv")

#-------------------------------------------------------------------------------------------------
#Handling missing values
#--------------------------------------------------------------------------------------------------
#https://www.kaggle.com/rtatman/data-cleaning-challenge-scale-and-normalize-data?utm_medium=email&utm_source=mailchimp&utm_campaign=5DDC-data-cleaning
#set seed for reproducibility
np.random.seed(0)

# look at a few rows of the nfl_data file. I can see a handful of missing data already!
print(inputData.sample(5))
inputData.head(5)
inputData.tail(5)

# get the number of missing data points per column
missing_values_count = inputData.isnull().sum()
print(missing_values_count)
# look at the # of missing points in the first ten columns
print(missing_values_count[0:10])

# how many total missing values do we have?
total_cells = np.product(inputData.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
missing_percentage = float((total_missing*100)/total_cells)
print(total_cells)
print(total_missing)
print(missing_percentage)

# remove all the rows that contain a missing value
rowsWithNullDropped=inputData.dropna()
# remove all the columns that contain a missing value
colsWithNullDropped=inputData.dropna(axis=1)

# just how much data did we lose?
print("Number of column in original data :%d" %inputData.shape[1])
print("Number of column in null values dropped :%d" %colsWithNullDropped.shape[1])

filledWithZero=inputData.fillna(0)
# replace all NA's the value that comes directly after it in the same column,
# then replace all the reamining na's with 0
filledWithValueszeros = inputData.fillna(method='bfill',axis=0).fillna(0)

#--------------------------------------------------------------------------------------------------
#Inconsistent Data Entry
#--------------------------------------------------------------------------------------------------

# look at the first ten thousand bytes to guess the character encoding
with open("C:\\Users\\pt2\\PycharmProjects\\Kaggle_ML_Cource\\data_cleaning_challenge\\All_Opp_27032018.csv",'rb') as rawdata:
    result=chardet.detect(rawdata.read(100))

# check what the character encoding might be
print(result)

# get all the unique values in the 'City' column
Op_Industry = inputData['Account Name: Industry/Vertical'].unique()
Op_Industry.sort()
print(Op_Industry)
#converting to lower case
lower_OP_owner=inputData['Account Name: Industry/Vertical'].str.lower()
#removing the trialing white spacees
trimmed_lower_OP_owner=lower_OP_owner.str.strip()
print(trimmed_lower_OP_owner)

matches=fuzzywuzzy.process.extract("Transportation",Op_Industry,limit=10000000,scorer=fuzzywuzzy.fuzz.token_sort_ratio)
print(matches)

# function to replace rows in the provided column of the provided dataframe
# that match the provided string above the provided ratio with the provided string
def replace_matches_column(df,column,string_to_match,min_ratio=51):
    strings=df[column].unique()#get a list of unque strings
    matches=fuzzywuzzy.process.extract(string_to_match,strings,limit=1000000,scorer=fuzzywuzzy.fuzz.token_sort_ratio)#get the matches for us string
    close_matches=[matches[0] for matches in matches if matches[1] >=min_ratio]
    rows_with_matches=df[column].isin(close_matches)
    df.loc[rows_with_matches,column]=string_to_match

replace_matches_column(df=inputData,column='Account Name: Industry/Vertical',string_to_match="Transportation")












