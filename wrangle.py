# =======================================================================================================
# Table of Contents START
# =======================================================================================================

'''
1. Orientation
2. Imports
3. acquire_zillow
4. prepare_zillow
5. wrangle_zillow
6. acquire_mall
7. prepare_mall
8. wrangle_mall
9. null_sum
10. drop_null
11. split
12. scale
13. sample_dataframe
14. remove_outliers
'''

# =======================================================================================================
# Table of Contents END
# Table of Contents TO Orientation
# Orientation START
# =======================================================================================================

'''
The purpose of this file is to create functions for the clustering machine learning process.
'''

# =======================================================================================================
# Orientation END
# Orientation TO Imports
# Imports START
# =======================================================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
import os
import env

# =======================================================================================================
# Imports END
# Imports TO acquire_zillow
# acquire_zillow START
# =======================================================================================================

def acquire_zillow():
    '''
    Obtains the vanilla version of zillow dataframe.
    '''
    query = '''
            SELECT
                *
            FROM 
                (SELECT 
                    parcelid, 
                    logerror, 
                    MAX(transactiondate) AS maxtransactiondate 
                FROM 
                    predictions_2017 
                WHERE 
                    transactiondate LIKE %s 
                GROUP BY 
                    parcelid, 
                    logerror) AS A
                LEFT JOIN properties_2017 USING(parcelid)
                LEFT JOIN airconditioningtype USING(airconditioningtypeid)
                LEFT JOIN architecturalstyletype USING(architecturalstyletypeid)
                LEFT JOIN buildingclasstype USING(buildingclasstypeid)
                LEFT JOIN heatingorsystemtype USING(heatingorsystemtypeid)
                LEFT JOIN propertylandusetype USING(propertylandusetypeid)
                LEFT JOIN storytype USING(storytypeid)
                LEFT JOIN typeconstructiontype USING(typeconstructiontypeid)
    '''
    params = ('2017%', )
    url = env.get_db_url('zillow')
    zillow = pd.read_sql(query, url, params=params)
    return zillow

# =======================================================================================================
# acquire_zillow END
# acquire_zillow TO prepare_zillow
# prepare_zillow START
# =======================================================================================================

def prepare_zillow():
    '''
    Takes the acquired zillow function and runs a set of prepatory calls for exploration.

    INPUT:
    NONE

    OUTPUT:
    zillow = pandas dataframe with prepared zillow data.
    '''
    zillow = acquire_zillow()
    zillow.dropna(subset=['latitude', 'longitude'], inplace=True)
    zillow = zillow[zillow.propertylandusedesc == 'Single Family Residential']
    zillow = drop_nulls(zillow, 0.25)
    drop_cols = [
        'propertylandusetypeid',
        'id',
        'calculatedbathnbr',
        'finishedsquarefeet12',
        'propertycountylandusecode',
        'rawcensustractandblock',
        'regionidcity',
        'regionidcounty',
        'regionidzip',
        'censustractandblock'
        ]
    zillow = zillow.drop(columns=drop_cols)
    mean_cols = [
        'calculatedfinishedsquarefeet',
        'fullbathcnt',
        'lotsizesquarefeet',
        'structuretaxvaluedollarcnt',
        'taxvaluedollarcnt',
        'landtaxvaluedollarcnt',
        'taxamount'
        ]
    mode_cols = [
        'yearbuilt'
        ]
    for col in mean_cols:
        zillow[col] = zillow[col].fillna(zillow[col].mean())
    for col in mode_cols:
        zillow[col] = zillow[col].fillna(zillow[col].mode()[0])
    return zillow

# =======================================================================================================
# prepare_zillow END
# prepare_zillow TO wrangle_zillow
# wrangle_zillow START
# =======================================================================================================

def wrangle_zillow():
    '''
    Acquires and prepares the zillow dataset then creates a .csv file if one does not exist and
    finally splits the dataframe.

    INPUT:
    NONE

    OUTPUT:
    zillow.csv = .csv file for more expedient usage
    train = train version of the prepped zillow dataframe
    validate = validate version of the prepped zillow dataframe
    test = test version of the prepped zillow dataframe
    '''
    if os.path.exists('zillow.csv'):
        zillow = pd.read_csv('zillow.csv', index_col=0)
        train, validate, test = split(zillow)
        return train, validate, test
    else:
        zillow = prepare_zillow()
        zillow.to_csv('zillow.csv')
        train, validate, test = split(zillow)
        return train, validate, test

# =======================================================================================================
# wrangle_zillow END
# wrangle_zillow TO acquire_mall
# acquire_mall START
# =======================================================================================================

def acquire_mall():
    '''
    Obtains the vanilla version of the mall dataframe.
    '''
    query = '''
            SELECT
                *
            FROM 
                customers
    '''
    url = env.get_db_url('mall_customers')
    mall = pd.read_sql(query, url)
    return mall

# =======================================================================================================
# acquire_mall END
# acquire_mall TO prepare_mall
# prepare_mall START
# =======================================================================================================

def prepare_mall():
    '''
    Acquires the vanilla mall dataframe and prepares it for later use for exploration.

    INPUT:
    NONE

    OUTPUT:
    mall = prepped version of the vanilla dataframe
    '''
    mall = acquire_mall()
    cols = [
        'age',
        'annual_income',
        'spending_score'
    ]
    mall = remove_outliers(mall, cols)
    dummies = pd.get_dummies(mall.select_dtypes(include='object'))
    mall = pd.concat([mall, dummies], axis=1)
    mall = mall.select_dtypes(exclude='object')
    return mall

# =======================================================================================================
# prepare_mall END
# prepare_mall TO wrangle_mall
# wrangle_mall START
# =======================================================================================================

def wrangle_mall():
    '''
    Acquires and prepares the mall dataset then creates a .csv file if one does not exist and
    finally splits the dataframe.

    INPUT:
    NONE

    OUTPUT:
    mall.csv = .csv file for more expedient usage
    train = train version of the prepped mall dataframe
    validate = validate version of the prepped mall dataframe
    test = test version of the prepped mall dataframe
    '''
    if os.path.exists('mall.csv'):
        mall = pd.read_csv('mall.csv', index_col=0)
        train, validate, test = split(mall)
        return train, validate, test
    else:
        mall = prepare_mall()
        mall.to_csv('mall.csv')
        train, validate, test = split(mall)
        return train, validate, test

# =======================================================================================================
# wrangle_mall END
# wrangle_mall TO null_sum
# null_sum START
# =======================================================================================================

def null_sum(df):
    '''
    Takes in a dataframe and returns a dataframe of the summarized nulls for each column.
    
    INPUT:
    df = pandas dataframe
    
    OUTPUT:
    nulls_df = pandas dataframe with summarized nulls of inputted dataframe
    '''
    df_features = df.columns.to_list()
    df_nullcnt = df.isna().sum().to_list()
    df_nullpct = round((df.isna().sum() / df.shape[0]), 4).to_list()
    temp_dict = {
    'feature_name' : df_features,
    'null_cnt' : df_nullcnt,
    'null_pct' : df_nullpct
    }
    nulls_df = pd.DataFrame(temp_dict).set_index('feature_name')
    return nulls_df

# =======================================================================================================
# null_sum END
# null_sum TO drop_nulls
# drop_nulls START
# =======================================================================================================

def drop_nulls(df, percent):
    '''
    Takes in a dataframe and a percent cutoff to return a dataframe with all the columns that are within the cutoff percentage.
    
    INPUT:
    df = pandas dataframe
    percent = Null percent cutoff. (0.00)
    
    OUTPUT:
    new_df = pandas dataframe with all columns that are within the cutoff percentage.
    '''
    original_cols = df.columns.to_list()
    drop_cols = []
    for col in original_cols:
        null_pct = df[col].isna().sum() / df.shape[0]
        if null_pct > percent:
            drop_cols.append(col)
    new_df = df.drop(columns=drop_cols)
    return new_df

# =======================================================================================================
# drop_nulls END
# drop_nulls TO split
# split START
# =======================================================================================================

def split(df):
    '''
    Takes a dataframe and splits the data into a train, validate and test datasets
    '''
    train_val, test = train_test_split(df, train_size=0.8, random_state=1349)
    train, validate = train_test_split(train_val, train_size=0.7, random_state=1349)
    print(f"train.shape:{train.shape}\nvalidate.shape:{validate.shape}\ntest.shape:{test.shape}")
    return train, validate, test


# =======================================================================================================
# split END
# split TO scale
# scale START
# =======================================================================================================

def scale(train, validate, test, cols, scaler):
    '''
    Takes in a train, validate, test and returns the dataframes,
    but scaled using the 'StandardScaler()'
    '''
    original_train = train.copy()
    original_validate = validate.copy()
    original_test = test.copy()
    scale_cols = cols
    scaler = scaler
    scaler.fit(original_train[scale_cols])
    original_train[scale_cols] = scaler.transform(original_train[scale_cols])
    scaler.fit(original_validate[scale_cols])
    original_validate[scale_cols] = scaler.transform(original_validate[scale_cols])
    scaler.fit(original_test[scale_cols])
    original_test[scale_cols] = scaler.transform(original_test[scale_cols])
    new_train = original_train
    new_validate = original_validate
    new_test = original_test
    return new_train, new_validate, new_test

# =======================================================================================================
# scale END
# scale TO sample_dataframe
# sample_dataframe START
# =======================================================================================================

def sample_dataframe(train, validate, test):
    '''
    Takes train, validate, test dataframes and reduces the shape to no more than 1000 rows by taking
    the percentage of 1000/len(train) then applying that to train, validate, test dataframes.

    INPUT:
    train = Split dataframe for training
    validate = Split dataframe for validation
    test = Split dataframe for testing

    OUTPUT:
    train_sample = Reduced size of original split dataframe of no more than 1000 rows
    validate_sample = Reduced size of original split dataframe of no more than 1000 rows
    test_sample = Reduced size of original split dataframe of no more than 1000 rows
    '''
    ratio = 1000/len(train)
    train_samples = int(ratio * len(train))
    validate_samples = int(ratio * len(validate))
    test_samples = int(ratio * len(test))
    train_sample = train.sample(train_samples)
    validate_sample = validate.sample(validate_samples)
    test_sample = test.sample(test_samples)
    return train_sample, validate_sample, test_sample

# =======================================================================================================
# sample_dataframe END
# sample_dataframe TO remove_outliers
# remove_outliers START
# =======================================================================================================

def remove_outliers(df, col_list, k=1.5):
    '''
    remove outliers from a dataframe based on a list of columns using the tukey method
    returns a single dataframe with outliers removed
    '''
    col_qs = {}
    for col in col_list:
        col_qs[col] = q1, q3 = df[col].quantile([0.25, 0.75])
    for col in col_list:
        iqr = col_qs[col][0.75] - col_qs[col][0.25]
        lower_fence = col_qs[col][0.25] - (k*iqr)
        upper_fence = col_qs[col][0.75] + (k*iqr)
        df = df[(df[col] > lower_fence) & (df[col] < upper_fence)]
    return df

# =======================================================================================================
# remove_outliers END
# =======================================================================================================