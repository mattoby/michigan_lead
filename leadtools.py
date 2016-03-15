#!/usr/local/bin/python

# for taxi data challenge
#import shapefile
import sys
#from dateutil.parser import parse as dateparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import time
from census import Census
from us import states
import os
import shapefile
import matplotlib.colors as colors
import pickle

# zipcodes:
from pyzipcode import ZipCodeDatabase

## sklearn imports:
#import sklearn.linear_model
#import sklearn.cross_validation
#from sklearn.linear_model import LogisticRegressionCV
from sklearn.grid_search import GridSearchCV
#from sklearn.cluster import KMeans
from sklearn.preprocessing import Imputer
import sklearn.linear_model as linear_model
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor

import xgboost as xgb

## mapping:
from mpl_toolkits.basemap import Basemap
import vincent
import shapefile
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

# import my modules:
sys.path.append('/Users/matto/Dropbox/Insight/datasciencetools/')
import datasciencetools as dst
import geotools as gt


# zcta = zipcode tabulation area:
def zcta_geographies(goodcoords=[-91.2, 40.8, -81.8, 48.2]):
    '''
    Defines the geographies of zip code tabulation areas (zcta's)
    in the united states.

    goodcoords = a coordinate set [lowlong, lowlat, hilong, hilat]
    outside of which a zcta will not be kept
    can be left blank (then all will be taken)
    The standard is the square surrounding Michigan.

    '''
    sf = shapefile.Reader('/Users/matto/Documents/censusdata/zipcodes/cb_2014_us_zcta510_500k/cb_2014_us_zcta510_500k')

    if len(goodcoords)>0:
        LOlong = goodcoords[0]
        LOlat = goodcoords[1]
        HIlong = goodcoords[2]
        HIlat = goodcoords[3]

    # Find zipcode and the shape of each zipcode:
    zctacodes = []
    zctashapes = []
    allrecords = sf.shapeRecords()
    for record in allrecords:
        zctacode = record.record[0]
        zctashape = record.shape

        if len(goodcoords)>0:
            lolong = record.shape.bbox[0]
            lolat = record.shape.bbox[1]
            hilong = record.shape.bbox[2]
            hilat = record.shape.bbox[3]
            condition = lolong>LOlong and lolat>LOlat and hilong<HIlong and hilat<HIlat
            if condition==True:
                zctacodes.append(zctacode)
                zctashapes.append(zctashape)

        else:
            zctacodes.append(zctacode)
            zctashapes.append(zctashape)



    return zctacodes, zctashapes


def import_lead_data_michigan():
    '''
    Import lead data for michigan
    note, I pre-added the zcta field. this can be altered if needed.
    '''
    fn_lead = '/Users/matto/Dropbox/Insight/interviews/arnhold/datachallenge/michegan_CLPPP_lead_bll_data_2013.csv'
    dfl = pd.read_csv(fn_lead)

    # create suffix for this table:
    dfl = dfl.add_suffix('__CLPPP')
    dfl=dfl.rename(columns = {'zip__CLPPP':'zip','zcta__CLPPP':'zcta'})

    return dfl


def census_filenames():

    fbase = '/Users/matto/Documents/censusdata/aff_download_byzip/data/'

    fnames = ['ACS_14_5YR_B02001_with_ann.csv',
    'ACS_14_5YR_S1701_with_ann.csv',
    'ACS_14_5YR_B02009_with_ann.csv',
    'ACS_14_5YR_S1702_with_ann.csv',
    'ACS_14_5YR_S0101_with_ann.csv',
    'ACS_14_5YR_S1901_with_ann.csv',
    'ACS_14_5YR_S0501_with_ann.csv',
    'ACS_14_5YR_S2301_with_ann.csv',
    'ACS_14_5YR_S0701_with_ann.csv',
    'ACS_14_5YR_S2502_with_ann.csv',
    'ACS_14_5YR_S0901_with_ann.csv',
    'ACS_14_5YR_S2503_with_ann.csv',
    'ACS_14_5YR_S1101_with_ann.csv',
    'ACS_14_5YR_S2504_with_ann.csv',
    'ACS_14_5YR_S1401_with_ann.csv',
    'ACS_14_5YR_S2506_with_ann.csv',
    'ACS_14_5YR_S1501_with_ann.csv',
    'ACS_14_5YR_S2507_with_ann.csv',
    'ACS_14_5YR_S1601_with_ann.csv',
    'ACS_14_5YR_S2701_with_ann.csv',
    'ACS_14_5YR_B25050_with_ann.csv',
    'ACS_14_5YR_DP04_with_ann.csv']

    # ftables (e.g., 'ACS_14_5YR_S1701_with_ann.csv'->'S1701'):
    ftables = []
    for fname in fnames:
        ftables.append(fname[11:-13])

    # fnames_meta (e.g., 'S1701'->'ACS_14_5YR_S1701_metadata.csv'):
    fnames_meta = []
    for ftable in ftables:
        fnames_meta.append('ACS_14_5YR_' + ftable + '_metadata.csv')

    return fbase, fnames, fnames_meta, ftables


def import_census_metadata(outfile='all_metadata.csv'):

    # import filenames:
    fbase, fnames, fnames_meta, ftables = census_filenames()

    # Create metadata dataframe for tables of interest:
    dfm = pd.DataFrame()
    for fn in fnames_meta:
        dfcurr = pd.read_csv(fbase + fn, header=None)
        dfcurr['fname'] = fn
        dfm = dfm.append(dfcurr)

    # Name columns and reindex:
    dfm.columns = ['code','desc','fname']
    dfm = dfm.reset_index(drop=True)

    # Add column for census data table filenames:
    # i.e., convert '_metadata.csv' to '_with_ann.csv' ending:
    dfm.loc[:,'fname_with_ann'] = np.vectorize(lambda fn: fn[:-13] + '_with_ann.csv')(dfm.loc[:,'fname'])

    # Add column for census data table names:
    dfm.loc[:,'tablename'] = np.vectorize(lambda fn: fn[11:-13])(dfm.loc[:,'fname'])

    # Add column for code__tablename:
    dfm.loc[:,'outcode'] = dfm['code'] + '__' + dfm['tablename']

    # Add coltype to dfm:
    dfm['coltype'] = 'census'

    # Make sure that outcode is unique:
    assert dfm['outcode'].unique().shape[0]==len(dfm),\
        'outcode should be unique'

    # Write the metadata dataframe to csv:
    if len(outfile)>0:
        dfm.to_csv(fbase + outfile)

    return dfm


def select_relevant_census_features_from_metadata(dfm):
    '''
    Chooses the metadata features that are relevant (first pass)
    These are ones that:
    (1) contain 'Percent'
    (2) contain 'Estimate'
    (3) do not contain 'PERCENT IMPUTED'
    '''

    # Create filters:
    percinds = np.vectorize(lambda desc: desc.find('Percent')>-1)(dfm['desc'])
    estinds = np.vectorize(lambda desc: desc.find('Estimate')>-1)(dfm['desc'])
    imputeinds = np.vectorize(lambda desc: desc.find('PERCENT IMPUTED')>-1)(dfm['desc'])
    yearbuiltinds = np.vectorize(lambda desc: desc.find('YEAR STRUCTURE BUILT')>-1)(dfm['desc'])
    marginerrinds = np.vectorize(lambda desc: desc.find('Margin of Error')>-1)(dfm['desc'])

    # filter1 (keep 'Percent', not 'Margin of Error', not 'PERCENT IMPUTED'):
    dfm1 = dfm.loc[percinds & ~marginerrinds & ~imputeinds, :]

    # filter2 (keep 'Estimate' and 'YEAR STRUCTURE BUILT'):
#    dfm2 = dfm.loc[estinds & yearbuiltinds, :]

    # combine:
#    dfm = dfm1.append(dfm2).drop_duplicates()
    dfm = dfm1

    # reindex:
    dfm = dfm.reset_index(drop=True)

    return dfm


def add_census_data_to_lead_data(dfl):
    '''
    Read in ACS census data from files
    The input is a 'dfm' dataframe, as defined in
    select_relevant_census_features_from_metadata()
    (it's a dataframe of metadata for these files, of the form:
     code, desc, fname
     HC03_EST_VC01, Percent below poverty.., ACS_14_5YR_S1701_metadata.csv)
    code is the column name
    desc is a description of the file
    fname is the filename holding that particular census table

    dfl is the lead dataframe. it is required, as the data from
    the census files pointed to by dfm
    ...
    '''

    # get filenames:
    fbase, fnames, fnames_meta, ftables = census_filenames()

    # import census metadata for columns of interest:
    dfm = import_census_metadata(outfile=[])
    dfm = select_relevant_census_features_from_metadata(dfm)

    # step through dfm, and pull out data:
    fnames_in_dfm = dfm['fname_with_ann'].unique()
    for fname in fnames_in_dfm:

        print 'processing file %s...' % fname

        # grab dataframe for current census table file:
        df_curr = pd.read_csv(fbase + fname, skiprows=[1], low_memory=False)

        # determine which columns to keep from census table:
        # 'GEO.id2' must be kept, since it has the zcta code.
        dfm_curr = dfm.loc[dfm['fname_with_ann']==fname,:]
        goodcols = dfm_curr.loc[:,'code'].values
        goodcols = np.append(goodcols, 'GEO.id2')
        zctacolind = np.where(goodcols=='GEO.id2')[0][0]

        # winnow down census table to only relevant columns:
        df_curr = df_curr.loc[:,goodcols]

        # add table name as suffix to all column titles:
        tablename = dfm.loc[dfm['fname_with_ann']==fname,:]['tablename'].values[0]
        df_curr = df_curr.add_suffix('__' + tablename)
        zctacol = df_curr.columns[zctacolind]

        # ensure that zip codes are all as integers:
        dfl.loc[:,'zcta'] = dfl['zcta'].astype(int)
        df_curr.loc[:,zctacol] = df_curr[zctacol].astype(int)

        # merge current census table into the main lead table:
        dfl = pd.merge(dfl, df_curr, how='left', left_on='zcta', right_on=zctacol)

    return dfl, dfm


def filter_extraneous_dfl_columns(dfl, toPlot=False):
    '''
    Filter out columns from the main dataframe that are not
    helpful for this task, or may be in error, have too many
    missing values, etc.
    '''

    print "Num dfl cols, pre filtering: %s" % dfl.shape[1]

    # Remove GEO.id2 columns (repeats of the zcta code):
    nonGEOcols = [col for col in dfl.columns if not ('GEO.id2' in col)]
    dfl = dfl[nonGEOcols]
    print "Num dfl cols, post-GEO.id2 filtering: %s" % dfl.shape[1]

    # Remove columns that are all nans:
    nonzerocols = [not np.isnan(m) for m in dfl.max()]
    dfl = dfl.loc[:,nonzerocols]
    print "Num dfl cols, post-allnan filtering: %s" % dfl.shape[1]

    # Remove columns with vals > 100 (since should be percents):
    # (only remove ones from table DP04, for now -- others are ok)
    nonDP04_gt100cols = [col for col in dfl.columns if not ('DP04' in col and dfl[col].max()>100)]
    dfl = dfl[nonDP04_gt100cols]
    print "Num dfl cols, post-perc>100 filtering: %s" % dfl.shape[1]

    # Remove columns with too few datapoints:
    if toPlot:
        print "Histogram of # non-nans in each column of the data"
        plt.figure()
        sns.distplot(dfl.count(),bins=40)
        plt.show()
    num_vals_cutoff = 800
    print 'Cutoff of # non-nan values: %s' % num_vals_cutoff
    goodcols = dfl.count()>num_vals_cutoff
    colstokeep = goodcols.loc[goodcols==True].index.values
    dfl = dfl[colstokeep]
    print "Num dfl cols, post-nan filtering: %s" % dfl.shape[1]

    return dfl


def align_dfm_to_dfl(dfm, dfl):

    '''
    Determine column types (i.e., features, etc.)
    dfm should be as output from add_census_data_to_lead_data
    will add the column types to dfm, and will make dfm match dfl
    columns
    '''

    # reset index in dfm:
    dfm = dfm.reset_index(drop=True)

    # Add non-census cols to dfm:
    colnames = dfl.columns
    dfmcols = dfm.columns.values
#    Ncols_dfm = len(dfm.columns)

    # loop through colnames, and populate a new metadata dataframe:
    dfm_new = pd.DataFrame(columns=dfmcols)
    for n, colname in enumerate(colnames):
        if any(dfm['outcode'].isin([colname])):
            dfm_new.loc[n] = dfm.loc[dfm['outcode']==colname,:].values[0]
        else:
            dfm_new.loc[n] = colname
            dfm_new.loc[n,'coltype'] = 'other'

    # check that all columns were addressed properly:
    assert all(dfm_new.outcode==dfl.columns), "some colnames in\
        dfl don't match rows in dfm. check code."

    # assign output:
    dfm = dfm_new

    return dfm


def filter_rows_missing_bllvals(dfl):
    print 'Filter for nan rows in bll counts:'
    print "Rows in dfl, pre-filtering: %s " % dfl.shape[0]
    dfl = dfl.loc[dfl['perc_bll_ge5__CLPPP'].notnull(),:]
    print "Rows in dfl, post-filtering bll>=5 rows: %s " % dfl.shape[0]
    dfl = dfl.loc[dfl['perc_bll_ge10__CLPPP'].notnull(),:]
    print "Rows in dfl, post-filtering bll>=10 rows: %s " % dfl.shape[0]
    print "\n'perc_bll_ge5' and 'perc_bll_ge10' are the lead features to predict.\n"

    return dfl


def keep_one_row_per_zcta(dfl):
    '''
    some zctas are repeated in dfl - this will take one row
    (the first) per zcta code.
    '''

    allzctas = dfl['zcta'].unique()
    dflcols = dfl.columns

    print 'Take 1 row per zcta (should combine - do later):'
    print "Rows in dfl, pre-filtering: %s " % dfl.shape[0]
    print "number of unique zctas: ", len(allzctas)

    dflnew = pd.DataFrame(columns=dflcols)
    for zcta in allzctas:
        dfcurr = dfl.loc[dfl['zcta']==zcta,:]
        if dfcurr.shape[0]>1:
            dfcurr = dfcurr.iloc[0,:]
        dflnew = dflnew.append(dfcurr)

    # reset index:
    dflnew.reset_index(drop=True)

    print "Rows in dfl, post-filtering: %s " % dflnew.shape[0]
    return dflnew


def build_michigan_lead_dataset(savetoFile=False):
    '''
    This will pull in the lead bll data, along with census data,
    to build a feature set that can hopefully predict chance of
    a child having lead exposure.
    '''

    # load in michegan lead bll data:
    dfl = import_lead_data_michigan()

    # add census data:
    dfl, dfm = add_census_data_to_lead_data(dfl)

    # convert non-numbers to nans:
    dfl = dfl.apply(pd.to_numeric, errors='coerce')

    # remove extraneous columns:
    dfl = filter_extraneous_dfl_columns(dfl, toPlot=True)

    # make the metadata dataframe align with the main dataframe:
    dfm = align_dfm_to_dfl(dfm,dfl)

    # filter out rows that have no value for %bll>5 or %bll>10:
    dfl = filter_rows_missing_bllvals(dfl)

    # filter rows so zctas are unique (there are a few with 2):
    dfl = keep_one_row_per_zcta(dfl)

    # save as csv:
    if savetoFile:
        fbase, fnames, fnames_meta, ftables = census_filenames()
        dfl.to_csv(fbase + 'michigan_lead_dataset.csv')

    return dfl, dfm


def feature_descriptions(dfm, features):
    '''
    Get description of a feature given the codename
    '''

    fdescs = []
    for feature in features:
        fdesc = dfm.loc[dfm['outcode']==feature,'desc'].iloc[0]
        fdescs.append(fdesc)

    fdescs = np.array(fdescs)

    return fdescs


def build_lasso(X_train, X_test, y_train, y_test):
    '''
    build linear regression lasso model
    '''
    # perform gridsearch to get best l1 parameter:
    params = {'alpha':[0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30,100,300,1000]}
    model = linear_model.Lasso()
    grid = GridSearchCV(model, params)
    grid.fit(X_train, y_train)
    #grid.fit(np.array([[1,2,3],[2,3,4],[5,3,1],[0,100,20]]),np.array([1,2,6,0]))

    # pick best model:
    print 'best params: %s' % grid.best_params_
    model = grid.best_estimator_
    y_pred = model.predict(X_test)
    importances = np.array(model.coef_)

    return model, y_pred, importances


def build_randforestregressor(X_train, X_test, y_train, y_test):
    '''
    build random forest regression model
    '''
    # perform gridsearch:
    params = {'max_depth':[None],'n_estimators':[10]}
    model = RandomForestRegressor()
    grid = GridSearchCV(model, params)
    grid.fit(X_train, y_train)

    # pick best model:
    print 'best params: %s' % grid.best_params_
    model = grid.best_estimator_

    y_pred = model.predict(X_test)
    importances = model.feature_importances_

    return model, y_pred, importances


def build_xgboostregressor(X_train, X_test, y_train, y_test):
    '''
    build xgboost regression model
    '''
    assert 1==0, 'need to get this code working after mvp..'

    # specify parameters via map, definition are same as c++ version
    param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'reg:linear'}
    # specify validations set to watch performance
    watchlist  = [(X_test,'eval'), (X_train,'train')]
    num_round = 2

    bst = xgb.train(param, X_train, num_round, watchlist)

    y_pred = model.predict(X_test)
    importances = model.feature_importances_

    return model, y_pred, importances


def build_gradientboostregressor(X_train, X_test, y_train, y_test):
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    importances = model.feature_importances_
    return model, y_pred, importances


def build_randforestclassifier(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=1000)
    model.fit(X_train, y_train)
    importances = model.feature_importances_
    y_pred_proba = model.predict_proba(X_test)[:,1]
    y_pred = model.predict(X_test)

    return model, y_pred, importances, y_pred_proba


def build_logreg(X_train, X_test, y_train, y_test):

    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    penalty = 'l1'
    cv = 10
    # run gridsearch:
    clf = GridSearchCV(LogisticRegression(penalty=penalty), param_grid=param_grid, cv=cv)
    clf = clf.fit(X_train, y_train)
    print 'best params: %s' % clf.best_params_
    model = clf.best_estimator_

    #model.fit(X_train, y_train) # needed?
    importances = np.array(model.coef_[0])
    y_pred_proba = model.predict_proba(X_test)[:,1]
    y_pred = model.predict(X_test)

    return model, y_pred, importances, y_pred_proba


def build_ML(dfl, dfm, X_names=[], y_name=[], modelType='lasso', MLtype='regression'):
    '''
    Builds a regression or classification model from the given
    lead bll data

    dfl: the feature dataframe (contains the y column too)
    dfm: metadata for the feature dataframe
    X_names: array of feature names (dfl cols) to go into model
    y_name: string denoting the dfl col that should be predicted

    Values are imputed for those missing in X

    If no X_names or y_name specified, default values are used
    '''

    # determine X columns:
    if X_names == []:
        X_names1 = dfm.loc[dfm['coltype']=='census','outcode'].values
        X_names2 = ['perc_pre1950_housing__CLPPP','children_under6__CLPPP']
        X_names3 = dfm.loc[dfm['coltype']=='close','outcode'].values
        X_names = np.append(X_names1,X_names2)
        X_names = np.append(X_names, X_names3)
    # remove any features that aren't in dfl.columns:
    X_names = np.array([f for f in X_names if f in dfl.columns])
#    X_names.shape

    print 'number of features in X: %s' % len(X_names)

    # determine y column:
    if y_name == []:
        y_name = 'perc_bll_ge5__CLPPP'

    # split into X and y:
    assert all([f in dfl.columns for f in X_names]), 'check cols'
    print ('children_under6__CLPPP' in dfl.columns)
    X = dfl.loc[:,X_names].values
    y = dfl.loc[:,y_name].values
    print 'shape of X: ', X.shape
    print 'shape of y: ', y.shape

    # split for cross validation:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 8)

    # scale features (skip for now, but must check for those that go above 100, and remove GEO.id2's!):

    # impute missing feature values:
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(X_train)
    X_train = imp.transform(X_train)
    X_test = imp.transform(X_test)
    X = imp.transform(X)

    # test that there are no nans left:
    assert sum(sum(np.isnan(X_train)))==0, 'nans in X_train'
    assert sum(sum(np.isnan(X_test)))==0, 'nans in X_test'
    assert sum(np.isnan(y_train))==0, 'nans in y_train'
    assert sum(np.isnan(y_test))==0, 'nans in y_test'

    # perform machine learning:
    if MLtype=='regression':
        if modelType=='lasso':
            model, y_pred, importances = build_lasso(X_train, X_test, y_train, y_test)
            y_pred_proba = []

        if modelType=='randomforestregressor':
            model, y_pred, importances = build_randforestregressor(X_train, X_test, y_train, y_test)
            y_pred_proba = []

        if modelType=='gradientboostregressor':
            model, y_pred, importances = build_gradientboostregressor(X_train, X_test, y_train, y_test)
            y_pred_proba = []

    elif MLtype=='classification':
        if modelType=='randomforestclassifier':
            model, y_pred, importances, y_pred_proba = build_randforestclassifier(X_train, X_test, y_train, y_test)

        if modelType=='logisticregression':
            model, y_pred, importances, y_pred_proba = build_logreg(X_train, X_test, y_train, y_test)

    else:
        print 'need to input a valid MLtype'

    # plot output:
    if MLtype=='regression':
        print 'R^2 for this model is: %s' % model.score(X_test, y_test)
        sns.regplot(y_test, y_pred)
        plt.xlabel('y test')
        plt.ylabel('y predicted')
        plt.show()

    if MLtype=='classification':
        dst.plot_roc_curve(y_test, y_pred_proba)

    # plot feature importances:
    nonzeroinds = np.where(np.abs(importances) > 0)[0]
    print 'Features with nonzero importances: ', len(nonzeroinds)
    Ximps = importances[nonzeroinds]
    Ximp_names = X_names[nonzeroinds]
    Ximp_descs = feature_descriptions(dfm, Ximp_names)
    dst.plot_feature_importances(Ximp_descs, Ximps,  Nfeatures=15)

    return model, importances, X, y, X_train, X_test, y_train, y_test, X_names, y_name, y_pred_proba


def get_zcta_centers(zctacodes, zctashapes):
    '''
    Get the (approx) centers of the zcta regions (lat and long)
    Returns a dataframe with lat and long columns
    zctacodes = list of zctacodes
    zctashapes = list of shape files of zctas
    '''

    centers_long = []
    centers_lat = []
    for zcta in zctashapes:
        center_long = np.mean([zcta.bbox[0], zcta.bbox[2]])
        center_lat = np.mean([zcta.bbox[1], zcta.bbox[3]])

        centers_long.append(center_long)
        centers_lat.append(center_lat)

    # create df of the zcta centers:
    df_zctacenters = pd.DataFrame({'longitude':centers_long,
                                  'latitude':centers_lat})
    df_zctacenters.index = zctacodes

    return df_zctacenters


def zipcode_cities(dfl):
    '''
    Find which city each zip code is in
    Not sure if this is zipcodes or zctas.. assume zipcodes
    '''

    zcdb = ZipCodeDatabase()

    dfl_cities = []
    for zipcode in dfl['zip']:
        try:
            city = zcdb[int(zipcode)].city
        except:
            city = 'other'
        dfl_cities.append(city)

    return dfl_cities


def zctas_for_dfl(zctacodes, zctashapes, dfl):
    '''
    This will find information on zctas lined up with rows of dfl
    There must be a 'zcta' column in dfl

    zctacodes = python list of zctacodes
    zctashapes = python list of shapefiles for each zcta
    dfl = the dataframe for lead data (or any df with a zcta col)
    '''

    # convert zctacodes to array of ints:
    zctacodes = np.array([int(code) for code in zctacodes])

    # get zctas from dfl:
    goodzctas = dfl['zcta']

    # Get shapefiles for each zcta code in dfl:
    dfl_zctashapes = []
    for zcta in goodzctas:
        assert zcta in zctacodes, 'zcta %s not in zctacodes.' % zcta
        goodrow = np.where(zctacodes==zcta)[0][0]
        currshape = zctashapes[goodrow]
        dfl_zctashapes.append(currshape)

    # get zcta shape centers:
    dfl_zctacenters = get_zcta_centers(dfl['zcta'], dfl_zctashapes)

    # get zcta cities:
    dfl_cities = zipcode_cities(dfl)

    return dfl_zctashapes, dfl_zctacenters, dfl_cities


def draw_zipcenters(zctacodes, zctashapes, dfl, colorcol='', gamma=1.0):
    dfl_zctashapes, dfl_zctacenters, dfl_cities = \
        zctas_for_dfl(zctacodes, zctashapes, dfl)

    # set style:
    sns.set(style="white", color_codes=True, font_scale=1.5)
#    sns.color_palette("Blues")

    # plot:
#    plt.figure()

    # determine what column to color by:
    if len(colorcol)>0:
        c = dfl[colorcol].copy()#**gamma
        plt.scatter(dfl_zctacenters['longitude'],\
                    dfl_zctacenters['latitude'],c=c, norm=colors.PowerNorm(gamma=gamma), cmap='Reds')
    else:
        plt.scatter(dfl_zctacenters['longitude'],\
                    dfl_zctacenters['latitude'])

    plt.colorbar()
#    plt.show()

    # return to default (this is a hack..)
    sns.set(style="darkgrid", color_codes=True, font_scale=1.5)

    return dfl_zctashapes, dfl_zctacenters


#def build_zip_distance_matrix(dfl_zctashapes, dfl, fromScratch=False, fnamewrite="dfl_mat_geodists.p", fnameread="dfl_mat_geodists.p"):
#    if fromScratch:
#        df_shapecenters, dfl_mat_geodists = gt.build_geodist_matrix(dfl['zcta'], dfl_zctashapes)
#        pickle.dump( dfl_mat_geodists, open( fnamewrite, "wb" ) )
#    else:
#        dfl_mat_geodists = pickle.load( open( fnameread, "rb" ) )
#
#    return dfl_mat_geodists


def find_neighboring_zctas(dfl, dfl_distmat, Nneighbors=5):
    '''
    Find the closest N zctas to each zcta in dfl.
    where N is Nneighbors
    '''

    zctas = dfl['zcta'].values
    dfl_closezctas = np.empty([len(zctas),Nneighbors])
    dfl_closezctadists = np.empty([len(zctas),Nneighbors])

    for n, zcta in enumerate(zctas):
        currdists = dfl_distmat[n,:]
        sortinds = np.argsort(currdists)

        # keep Nneighbors+1 (since dist to self will be 0):
        goodinds = sortinds[:Nneighbors+1]
        # now, remove self from results:
        currindplace = np.where(n==goodinds)[0][0]
        goodinds = np.delete(goodinds,currindplace)
        # check that the length of the result is correct:
        assert len(goodinds)==Nneighbors, 'problem with code..'

        closedists = currdists[goodinds]
        closezctas = zctas[goodinds]

        # populate output:
        dfl_closezctas[n,:] = closezctas
        dfl_closezctadists[n,:] = closedists

    return dfl_closezctas, dfl_closezctadists


def build_neighboring_zcta_features(dfl, dfm, dfl_closezctas):
    '''
    Create new feature, which are averaged features from the
    zcta areas surrounding the current zcta area.

    Note: dfl, dfm, and dfl_closezctas must line up!
    '''

    # determine feature sets to average:
    features1 = dfm.loc[dfm['coltype']=='census','outcode'].values
    features2 = ['perc_pre1950_housing__CLPPP']
    features = np.append(features2,features1)

    # create new feature dataframe (to be merged with dfl):
    dflnew = pd.DataFrame(columns=features)

    # form new features in dflnew for the given row of dfl:
    allzctas = dfl['zcta'].values
    for n, zcta in enumerate(allzctas):

        # get the zctas that are close to the current zcta:
        close_zctas = dfl_closezctas[n,:]
        # which rows of dfl have zctas close to the current zcta:
        closerows = [zcta in close_zctas for zcta in dfl['zcta']]
        assert sum(closerows)==5, 'there should be 5 close zctas..'
        # take the means of those rows:
        dfl_close = dfl.loc[closerows,features]
        dfl_close_means = dfl_close.mean()
        # populate dflnew with these mean values:
#        np.append(dfl_close_means, zcta)
        dflnew = dflnew.append(dfl_close_means, ignore_index=True)
#        dflnew.iloc[n,:] = dfl_close_means

    # add suffix to dflnew:
    dflnew = dflnew.add_suffix('_close')

    # fix indexes for concatenation:
    dfl = dfl.reset_index(drop=True)
    dflnew = dflnew.reset_index(drop=True)
    assert all(dfl.index==dflnew.index), "indexes don't match."

    # update dfl with new features:
    dfl2 = pd.concat([dfl, dflnew], axis=1)
    assert dfl2.shape[0]==dfl.shape[0], 'num rows changed - check'

    # create new part of dfm:
    gooddfmrows = [f in features for f in dfm['outcode']]
    dfmnew = dfm.loc[gooddfmrows,:]
    dfmnew.loc[:,'coltype'] = 'close'
    dfmnew.loc[:,'outcode'] = [oc + '_close' for oc in dfmnew['outcode']]
    dfmnew.loc[:,'desc'] = [desc + '; close zctas' for desc in dfmnew['desc']]

    # merge dfmnew with dfm:
    dfm2 = dfm.append(dfmnew)

    # check that the shapes are correct, and columns line up:
    assert dfm2.shape[0]==dfl2.shape[1], 'check shapes of dfm and dfl'
    assert all(dfm2['outcode']==dfl2.columns), 'check cols in dfl'

    print 'Note: only desc, outcode, and coltype are updated in dfm'

    return dfl2, dfm2

########

# y = aggregate_prediction(X, y, model_class, model_regress)
# predict if *any* bll>=5
# if so, predict how much




















