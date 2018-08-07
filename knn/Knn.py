# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

# Read csv file into a dataframe
def initialize_from_file(filename):
    """
    initialize this example set from a file as specified by the exercise
    """
    df = pd.read_csv(filename)
    return df

# Initialized so they can be used
row = 0
column = 0
row_tst = 0
column_tst = 0

# This is a print function to know the progress.
def progress(x):
    out = '%s is the current iteration' % x  # The output
    bs = '\b' * 1000            # The backspace
    print(bs)
    print(out)

# Data is read and normalized here.
def loadDataset(trainFilename, testFilename):
    global row, column, row_tst, column_tst
    #Training and test csv file read and normalize
	#Traning data normalization
    df = initialize_from_file(trainFilename)
    df_tst = initialize_from_file(testFilename)
    df = df.astype(float)
    row, column = df.axes
    kmean = []
    kstd = []
    # For 
    for k in range(len(column) - 1):
        kmean.append(df[:][column[k]].mean())
        kstd.append(df[:][column[k]].std())

    for k in range(len(column) - 1):
        for r in row:
            df.at[r,column[k]] = (df.at[r , column[k]] - kmean[k])/kstd[k]
    print("-----------NORMALIZED TRAINING DATA--------------")
    row_tst, column_tst = df_tst.axes
    for k in range(len(column_tst) - 1):
        for r in row_tst:
            df_tst.at[r,column_tst[k]] = (df_tst.at[r , column_tst[k]] - kmean[k])/kstd[k]
    print("-----------NORMALIZED Test DATA--------------")
    return df, df_tst

# Cosine similarity between vectors found
def cosine_similarity(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    v1_np = v1.values
    v2_np = v2.values
    return v1_np.dot(v2_np)/np.sqrt(v1_np.dot(v1_np) * v2_np.dot(v2_np))
# =============================================================================
#     for i in range(len(v1)):
#         x = v1[i]; y = v2[i]
#         sumxx += x*x
#         sumyy += y*y
#         sumxy += x*y
#     return sumxy/math.sqrt(sumxx*sumyy)
# =============================================================================

# Output is predicted based on the training data. This is done for K = 1, 3 and 5
def Predict_output(df, df_tst):
    global row, column, row_tst, column_tst
    
    match_k1 = 0
    match_k3 = 0
    match_k5 = 0
    num_of_rows = len(row_tst)
    for trn_data in range(num_of_rows):
        match_k3_ = 0
        match_k5_ = 0
        progress(trn_data)
        cos_sim = []
        for r in range(len(row)):
            cos_sim.append(cosine_similarity(df_tst.iloc[trn_data,0:(len(column)-1)], df.iloc[r,0:(len(column_tst) - 1)]))

        pos1 = cos_sim.index(max(cos_sim))
        cos_sim[pos1] = -1

        pos2 = cos_sim.index(max(cos_sim))
        cos_sim[pos2] = -1

        pos3 = cos_sim.index(max(cos_sim))
        cos_sim[pos3] = -1

        pos4 = cos_sim.index(max(cos_sim))
        cos_sim[pos4] = -1

        pos5 = cos_sim.index(max(cos_sim))

        if(df.iloc[pos1, -1] == df_tst.iloc[trn_data, -1]):
            match_k1 += 1
            match_k3_ += 1
            match_k5_ += 1

        if(df.iloc[pos2, -1] == df_tst.iloc[trn_data, -1]):
            match_k3_ += 1
            match_k5_ += 1

        if(df.iloc[pos3, -1] == df_tst.iloc[trn_data, -1]):
            match_k3_ += 1
            match_k5_ += 1

        if(df.iloc[pos4, -1] == df_tst.iloc[trn_data, -1]):
            match_k5_ += 1

        if(df.iloc[pos5, -1] == df_tst.iloc[trn_data, -1]):
            match_k5_ += 1
        
        if(match_k3_ >= 2):
            match_k3 += 1
        
        if(match_k5_ >= 3):
            match_k5 += 1

    print('With k = 1, The match percentage is', match_k1/num_of_rows)
    print('With k = 3, The match percentage is', match_k3/num_of_rows)
    print('With k = 5, The match percentage is', match_k5/num_of_rows)
# Function call for data update and Normalization.
df, df_tst = loadDataset('trainingData_.csv', 'testData_.csv')
# Prediction for test data.
Predict_output(df, df_tst)


# print('Train: ' + repr(len(trainingSet)))
# =============================================================================
