import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from collections import defaultdict, Counter
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from operator import itemgetter
from sklearn.svm import SVC
import warnings
from sklearn.neural_network import MLPClassifier
from statistics import stdev,mean,median


def split_data(full,past_n_week = False,num_past_weeks=9,curr_year=2019):
#     if 'Money_home' in list(full.columns):
#         full = full.drop('Money_home',axis = 1)
    if past_n_week:
        test_weeks = list(full[full.schedule_season == curr_year]['schedule_week'].unique())[-1*(num_past_weeks):]
        validation = full[(full.schedule_season == curr_year)&(full.schedule_week.isin(test_weeks))]
        yvalid = validation[['spread_target']]
        xvalid = validation.drop('spread_target',axis = 1)
        avoid = list(xvalid.game_info.tolist())
    else:
        validation = full[(full.schedule_season > curr_year - 2)]
        yvalid = validation[['spread_target']]
        xvalid = validation.drop('spread_target',axis = 1)
        avoid = list(xvalid.game_info.tolist())
        
    train_set = full[~full.game_info.isin(avoid)]
    ytrain = train_set[['spread_target']]
    xtrain = train_set.drop('spread_target',axis=1)
    return xtrain,ytrain,xvalid,yvalid
    
def scale_df(X,test=False):
    if test:
        X.drop('score_differential',axis=1,inplace=True)
    dont_touch =  ['game_info','streak_diff','spread_col','spread_target']\
                            + [i for i in X.columns if 'schedule' in i]\
                            + [i for i in X.columns if 'stadium' in i]
    if 'hometeam_win' in list(X.columns):
        dont_touch.append('hometeam_win')
    if test:
        dont_touch.remove('spread_target')
    to_use = [i for i in X.columns if i not in dont_touch]
    scaled=pd.DataFrame(scale(X[to_use]))
    new_df = pd.concat([X[dont_touch],scaled],axis = 1)
    new_df.columns = dont_touch + to_use
    return new_df
def prelim_selection(X,Y,l_of_l):
    model = XGBClassifier(gamma = 0.5,subsample = 0.6)
    sel_ = SelectFromModel(model,max_features = 1)
    chosen_ones = []
    for group in l_of_l:
        sel_.fit(X[group],Y)
        selected_feat = X[group].columns[(sel_.get_support())].values
        chosen_ones.append(selected_feat)
    chosen_ones = [i.tolist()[0] for i in chosen_ones]
    return chosen_ones
def find_top_indices(data,top):
    sorted_data = sorted(enumerate(data), key=itemgetter(1),reverse=True)
    return [d[0] for d in sorted_data[:top]]

def top_vars(X,Y,prelim_cols,topn):
    X = X[prelim_cols]
    model = XGBClassifier(gamma = 0.5,subsample = 0.6)
    model.fit(X,Y,eval_metric = 'error', verbose = 0)
    diction = dict(zip(X.columns,model.feature_importances_))
    diction = {k:v for k,v in diction.items() if v > 0}
    indices = find_top_indices(list(diction.values()),top=topn)
    selected_columns = [list(diction.keys())[i] for i in indices]
    return selected_columns

def start_col(df):
    for i in df.columns:
        if 'DIFF' in i:
            return list(df.columns).index(i)

def process_df(fp='/Users/Ben Rosen/Desktop/pwd_test25/full_training.csv',test=False,ml = False):

    full = pd.read_csv(fp,encoding = 'iso-8859-1')
    if not test:
        full = full[full.spread_target.notnull()]
    full['schedule_week'] = full.schedule_week.apply(int)
    year_cols = pd.get_dummies(full.schedule_season)

    if test:
        year_cols.columns = ['year2019']
    
    else:
        year_cols.columns = ['year{}'.format(i) for i in range(2010,2020)]
    full = pd.concat([full,year_cols],axis=1)
    
    full['Spread_home'] = full.Spread_home.str.replace('%','').apply(float)
    full['Money_home'] = full.Money_home.replace('true_null',np.nan)
    full.loc[full.Money_home.notnull(),'Money_home'] = full.Money_home.str.replace('%','').apply(float)
    full['O/U_home'] = full['O/U_home'].replace('true_null',np.nan)
    full.loc[full['O/U_home'].notnull(),'O/U_home'] = full['O/U_home'].str.replace('%','').apply(float)
    full['game_info'] = list(zip(full.team_home,full.team_away,full.schedule_season,full.schedule_week))
    full['schedule_season'] = full.schedule_season.astype(int)
    full['schedule_week'] = full.schedule_week.astype(int)
    yml = full[['hometeam_win']]
    year_cols = [i for i in full.columns if i.startswith('year')]
    cols_to_drop = ['spread_favorite','over_under_line','schedule_playoff',
'weather_humidity','missing_weather','weather_temperature','weather_wind_mph','zipper_touse','modified_home_names',
'modified_away_names','schedule_date','team_home','stadium','team_away','schedule_date','hometeam_win','score_differential']
    front_cols = ['game_info','streak_diff','spread_col','spread_target',
                 'schedule_season','schedule_week','Spread_home','Money_home','O/U_home'] + year_cols
    if test:
        cols_to_drop.remove('score_differential')
        front_cols.remove('spread_target')
    if ml:
        cols_to_drop.remove('hometeam_win')
        front_cols.append('hometeam_win')
    full.drop(cols_to_drop,axis=1,inplace=True)
    ao_cols = [i for i in full.columns if i not in front_cols]
    full = pd.concat([full[front_cols],full[ao_cols]],axis=1)
    full =full.reset_index(drop=True)

    if test:
        full = scale_df(full,test=True)
        return full

    else:
        full = scale_df(full,test=False)
        fully = full[['spread_target']]
        fullx = full.drop('spread_target',axis=1)
        return full,fullx,fully

def col_groups(df,start_column):
    dd = defaultdict(list)
    possibs = ['home_','away_','current_year_','last_1_','last_3_','last_year_']
    for col in list(df.columns)[start_column:]:
        for prefx in possibs:
            if prefx in col[:17]:
                strt = col.index(prefx) + len(prefx)
        end = col.index('_DIFF')
        key = col[strt:end]
        dd[key].append(col)
    return list(dd.values())
