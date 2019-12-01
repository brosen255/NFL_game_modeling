import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression,RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn import metrics
from sklearn.metrics import accuracy_score
from statistics import mean,median
import warnings
warnings.filterwarnings("ignore")

def data_wrangler(iterative_df,score_weather,curr_week,test):
    print('turning up the jets...')

    #path,filename,ext = '/Users/Ben/Documents/pwd_scraping/', 'spreadspoke_scores','.csv'
    filepath = '/Users/Ben/Desktop/pwd_test/'
    filename,ext = 'spreadspoke_scores','.csv'
    game_data_since_66 = pd.read_csv(filepath + filename + ext)

    abb = pd.read_csv(filepath + 't_abb.csv')
    abb_dict = dict(zip(abb[abb.columns[0]],abb[abb.columns[1]]))

    cutoff_year = 2003
    o_three = game_data_since_66[game_data_since_66['schedule_season'] >= cutoff_year]
    o_three ['team_home1'] = o_three.team_home.str.replace('St. Louis Rams','Rams')
    o_three ['team_away1'] = o_three.team_away.str.replace('St. Louis Rams','Rams')
    o_three ['team_home1'] = o_three.team_home1.str.replace('Los Angeles Rams','Rams')
    o_three ['team_away1'] = o_three.team_away1.str.replace('Los Angeles Rams','Rams')
    o_three ['team_home1'] = o_three.team_home1.str.replace('San Diego Chargers','Chargers')
    o_three ['team_away1'] = o_three.team_away1.str.replace('San Diego Chargers','Chargers')
    o_three ['team_home1'] = o_three.team_home1.str.replace('Los Angeles Chargers','Chargers')
    o_three ['team_away1'] = o_three.team_away1.str.replace('Los Angeles Chargers','Chargers')
    o_three['spread_favorite_team'] = o_three['team_favorite_id'].map(abb_dict)
    o_three.loc[o_three['spread_favorite_team'] == o_three.team_away1,'spread_favorite'] = o_three.spread_favorite.apply(lambda x: x*(-1))
    o_three.drop(['spread_favorite_team','team_home1','team_away1'], axis = 1, inplace = True)

    thresh_date = o_three[o_three.columns[:3]].drop_duplicates(subset = ['schedule_season','schedule_week'], keep = 'last')
    thresh_date = thresh_date.reset_index().drop('index',axis = 1)
    thresh_date['week_sep'] = pd.to_datetime(thresh_date['schedule_date'])
    thresh_date['week_sep'] = pd.DatetimeIndex(thresh_date.week_sep) + pd.DateOffset(1)
    ll = [str(i) for i in thresh_date.week_sep.tolist() ]
    ll = [i[:10] for i in ll]
    date_to_week_dict = dict(zip(ll,thresh_date.schedule_week))
    print(iterative_df.columns)
    
    if not [i for i in iterative_df.columns if 'date' in i]:
        iterative_df['date'] = list(iterative_df.reset_index(level=['date'])['date'])

    iterative_df['week'] = iterative_df['date'].map(date_to_week_dict)
    iterative_df['week'] = iterative_df.week.str.replace('SuperBowl','Superbowl')
    iterative_df['week'] = iterative_df.week.str.replace('WildCard','Wildcard')

    iterative_df['week'] = iterative_df.week.str.replace('Wildcard','18')
    iterative_df['week'] = iterative_df.week.str.replace('Division','19')
    iterative_df['week'] = iterative_df.week.str.replace('Conference','20')
    iterative_df['week'] = iterative_df.week.str.replace('Superbowl','21')

    iterative_df['week'] = iterative_df.week.apply(lambda x: float(x) + 1)

    iterative_df['year'] = iterative_df.date.apply(str).apply(lambda x: x.split('-')[0])

    o_three['team_home'] = o_three['team_home'].str.replace('St. Louis Rams','Los Angeles Rams')
    o_three['team_home'] = o_three['team_home'].str.replace('San Diego Chargers','Los Angeles Chargers')
    o_three['team_away'] = o_three['team_away'].str.replace('St. Louis Rams','Los Angeles Rams')
    o_three['team_away'] = o_three['team_away'].str.replace('San Diego Chargers','Los Angeles Chargers')

    print('matching team names between datasets')
    nc = {}
    nc['Washington Redskins'] = 'Washington'
    nc['Buffalo Bills'] = 'Buffalo'
    nc['Carolina Panthers'] = 'Carolina'
    nc['Cincinnati Bengals'] = 'Cincinnati'
    nc['Cleveland Browns'] = 'Cleveland'
    nc['Dallas Cowboys'] = 'Dallas'
    nc['Detroit Lions'] = 'Detroit'
    nc['Green Bay Packers'] = 'Green Bay'
    nc['Kansas City Chiefs'] = 'Kansas City'
    nc['Miami Dolphins'] = 'Miami'
    nc['New York Giants'] = 'NY Giants'
    nc['Pittsburgh Steelers'] = 'Pittsburgh'
    nc['San Francisco 49ers'] = 'San Francisco'
    nc['Seattle Seahawks'] = 'Seattle'
    nc['Tennessee Titans'] = 'Tennessee'
    nc['Philadelphia Eagles'] = 'Philadelphia'
    nc['Arizona Cardinals'] = 'Arizona'
    nc['Atlanta Falcons'] = 'Atlanta'
    nc['Baltimore Ravens'] = 'Baltimore'
    nc['Indianapolis Colts'] = 'Indianapolis'
    nc['Jacksonville Jaguars'] = 'Jacksonville'
    nc['Minnesota Vikings'] = 'Minnesota'
    nc['New Orleans Saints'] = 'New Orleans'
    nc['New York Jets'] = 'NY Jets'
    nc['Oakland Raiders'] = 'Oakland'
    nc['Los Angeles Chargers'] = 'LA Chargers'
    nc['Los Angeles Rams'] = 'LA Rams'
    nc['Tampa Bay Buccaneers'] = 'Tampa Bay'
    nc['Houston Texans'] = 'Houston'
    nc['New England Patriots'] = 'New England'
    nc['Denver Broncos'] = 'Denver'
    nc['Chicago Bears'] = 'Chicago'

    o_three['hometeam_clean'] = o_three['team_home'].map(nc)
    o_three['awayteam_clean'] = o_three['team_away'].map(nc)

    final_stats = iterative_df.copy()

    left_width = o_three.shape[1]
    right_width = final_stats.shape[1]

    final_stats['week'] = final_stats.week.apply(str).str.replace('SuperBowl','Superbowl').str.replace('WildCard','Wildcard')
    final_stats['week'] = final_stats.week.apply(str).str.replace('Wildcard','18.1').str.replace('Division','19.1')
    final_stats['week'] = final_stats.week.apply(str).str.replace('Conference','20.1').str.replace('Superbowl','21.1')
    final_stats['week'] = final_stats.week.apply(float)


    o_three['schedule_week'] = o_three['schedule_week'].apply(str).str.replace('SuperBowl','Superbowl').str.replace('WildCard','Wildcard')
    o_three['schedule_week'] = o_three['schedule_week'].apply(str).str.replace('Wildcard','18').str.replace('Division','19')
    o_three['schedule_week'] = o_three['schedule_week'].apply(str).str.replace('Conference','20').str.replace('Superbowl','21')
    o_three['schedule_week'] = o_three['schedule_week'].apply(float)
    o_three['schedule_season'] = o_three['schedule_season'].apply(str)

    final_stats['zipper'] = list(zip(final_stats['team'],final_stats['year'],final_stats['week']))
    o_three['zipper_home'] = list(zip(o_three['hometeam_clean'],o_three['schedule_season'],o_three['schedule_week']))
    o_three['zipper_away'] = list(zip(o_three['awayteam_clean'],o_three['schedule_season'],o_three['schedule_week']))

    home_merger = final_stats.copy()
    home_merger.columns = [i+'_home' for i in home_merger.columns]
    away_merger = final_stats.copy()
    away_merger.columns = [i+'_away' for i in away_merger.columns]

    total_merger = pd.merge(o_three,home_merger,how = 'left',on = 'zipper_home', indicator = True)
    total_merger = total_merger[total_merger['_merge'] == 'both']
    tot_merg_cols = list(total_merger.columns)
    end_col_ = [i for i in tot_merg_cols if 'last_year_opponent-penalties-per-play_home' in i][0]
    end_idx = tot_merg_cols.index(end_col_) + 1
    total_merger = total_merger[tot_merg_cols[:end_idx]]

    total_merger = pd.merge(total_merger,away_merger,how = 'left',on = 'zipper_away', indicator = True)
    total_merger = total_merger[total_merger['_merge'] == 'both']
    tot_merg_cols = list(total_merger.columns)
    end_col_ = [i for i in tot_merg_cols if 'last_year_opponent-penalties-per-play_away' in i][0]
    end_idx = tot_merg_cols.index(end_col_) + 1
    total_merger = total_merger[tot_merg_cols[:end_idx]]

    #total_merger = total_merger[total_merger.schedule_season != '2019']
    total_merger = total_merger[total_merger.schedule_week != 1]

    tot_merg_cols = list(total_merger.columns)
    start_col_ = [i for i in tot_merg_cols if 'away_points-per-game_home' in i][0]
    start_idx = tot_merg_cols.index(start_col_)
    print('Start index should be ~ 23: ', start_idx)

    for col in total_merger.columns[start_idx:]:
            total_merger[col] = total_merger[col].apply(str).str.replace('--','0')

    team_cols2 = [i for i in total_merger.columns if 'eam' in i if 'special' not in i if 'last' not in i if 'aver' not in i]
    date_cols2 = [i for i in total_merger.columns if 'date' in i]
    cols_to_impute = [i for i in total_merger.columns[start_idx:] if i not in team_cols2 if i not in date_cols2]

    print('PROGRESS FOR IMPUTING MISSING VALUES: ')
    for col in cols_to_impute:
        progress = str(round(cols_to_impute.index(col) / len(cols_to_impute) * 100))
        if progress == '0':
            if cols_to_impute.index(col) == 0:
                stored_new_multiple_of_10 = '0'
                print( stored_new_multiple_of_10 + '%' )
        else:
            if (progress[-1] == '0') & (progress != stored_new_multiple_of_10):
                stored_new_multiple_of_10 = progress
                print( stored_new_multiple_of_10 + '%' )

        if total_merger[total_merger[col].apply(str).str.contains(':')].shape[0] != 0:
            notnull = total_merger[total_merger[col].notnull()][col].str.replace(':','.').apply(float).tolist()
            col_median = median(notnull)
            total_merger[col] = total_merger[col].fillna(col_median)
            total_merger[col] = total_merger[col].str.replace(':','')

        if total_merger[total_merger[col].apply(str).str.contains('%')].shape[0] != 0:
            notnull = total_merger[total_merger[col].notnull()][col].str.replace('%','').apply(float).tolist()
            col_median = median(notnull)
            total_merger[col] = total_merger[col].fillna(col_median)
            total_merger[col] = total_merger[col].str.replace('%','')

        else:
            notnull = total_merger[total_merger[col].notnull()][col].apply(str).str.replace(':','.').apply(float).tolist()
            col_median = median(notnull)
            total_merger[col] = total_merger[col].fillna(col_median)

    end_col_number = list(total_merger.columns).index('last_year_opponent-penalties-per-play_home')

    #width_of_stats = len(home_merger.columns) - 4
    template_df = total_merger[total_merger.columns[:start_idx]]

    print('calculating differentials for each stat')
    print('checking twice if subtracting the correct columns:')

    #end_col_number = list(total_merger.columns).index('last_year_opponent-penalties-per-play_away')
    for i in range(start_idx,end_col_number):
        curr_col = total_merger.columns[i]
        curr_col = curr_col.replace('_home','_DIFF')
        home_column = total_merger.columns[i]
        away_column = total_merger.columns[i + end_col_number - 20]

        if i == 50 or i == 1000:
            if home_column[:-5] != away_column[:-5]:
                raise ValueError('the home/away column are not matching up')
            else:
                print('columns match up')
        template_df[curr_col] = total_merger[home_column].apply(float) - total_merger[away_column].apply(float)
    print('creating binary column for hometeam win/loss...')
    template_df['hometeam_win'] = ''
    template_df.loc[template_df['score_home'] > template_df['score_away'],'hometeam_win'] = 1
    template_df.loc[template_df['score_home'] <= template_df['score_away'],'hometeam_win'] = 0

    template_df.loc[template_df.weather_detail.isnull(),'weather_detail'] = 'missing_weather'

    if template_df.shape[0] != pd.get_dummies(template_df['weather_detail']).shape[0]:
        raise ValueError('encountered errror when adding the weather detail variables to dataset')

    dummied_weather_detail = pd.get_dummies(template_df.weather_detail)
    new_total_cols = list(template_df.columns) + list(dummied_weather_detail.columns)
    template_df1 = pd.concat([template_df,dummied_weather_detail], ignore_index = True, axis = 1)
    template_df1.columns = new_total_cols
    template_df = template_df1.drop('weather_detail',axis = 1)

    still_null = [i for i in template_df.columns if template_df[i].isnull().any()]
    print('filling null values')
    for col in still_null:
        template_df[col] = template_df[col].fillna(template_df[col].median())


    team_cols2 = [i for i in template_df.columns if 'eam' in i if 'special' not in i if 'last' not in i if 'aver' not in i]
    date_cols2 = [i for i in template_df.columns if 'zipper' in i]
    date_cols2 += [i for i in template_df.columns if 'date' in i]
    date_cols2 += [i for i in template_df.columns if i == 'hometeam_win']
    date_cols2 += [i for i in template_df.columns if i == 'score_home']
    date_cols2 += [i for i in template_df.columns if i == 'score_away']
    predictor_cols = [i for i in template_df.columns if i not in team_cols2 if i not in date_cols2]

    print('Reading in stadium info...')
    stadium = pd.read_csv(filepath + 'nfl_stadiums.csv', encoding = 'iso-8859-1')
    stadium = stadium[['stadium_name','stadium_type','stadium_capacity','stadium_surface']]
    cap_list = stadium[stadium['stadium_capacity'].notnull()]['stadium_capacity'].tolist()
    stadium['stadium_capacity'] = stadium.stadium_capacity.fillna(median(cap_list)).str.replace(',','')
    stad_dumm = pd.get_dummies(stadium[['stadium_type','stadium_surface']],dummy_na = True)
    stadium_df = pd.concat([stadium['stadium_name'],stadium['stadium_capacity'],stad_dumm], axis = 1)
    X = pd.merge(template_df[predictor_cols],stadium_df,how = 'left',left_on = 'stadium',right_on = 'stadium_name').drop(['stadium','stadium_name'],axis = 1)
    med_cap = X[X['stadium_capacity'].notnull()]['stadium_capacity'].tolist()
    med_cap = [int(i) for i in med_cap]
    X.loc[X['stadium_capacity'].isnull(),'stadium_capacity'] = median(med_cap)

    stad_null = [i for i in stadium.columns if stadium[i].isnull().any()]
    null_nan = [i for i in stad_null if 'nan' in i]
    print('adding variables about each stadium and filling in missing values....')
    for col in null_nan:
        X[col] = X[col].fillna(1)

    still_null = [i for i in X.columns if X[i].isnull().any()]
    for col2 in still_null:
        X.loc[X[col2].isnull(),col2] = 0

    game_info = template_df[['team_home_x','team_away_x','schedule_date','schedule_season','schedule_week']]
    concat_cols = list(game_info.columns) + list(X.columns)
    Xtester = pd.concat([game_info,X],ignore_index=True,axis = 1)
    Xtester.columns = concat_cols

    game_data_since_66['stadium_zipper'] = list(zip(game_data_since_66.schedule_date,
                                           game_data_since_66.team_home,
                                           game_data_since_66.team_away))

    Xtester['stadium_zipper'] = list(zip(Xtester.schedule_date,
                                Xtester.team_home_x,
                                Xtester.team_away_x))
    Xtest = pd.merge(Xtester,game_data_since_66[['stadium_zipper','stadium']],how = 'left',on = 'stadium_zipper').drop('stadium_zipper',axis = 1)
    temp_df = Xtest['schedule_week']
    temp_df.columns = ['schedule_week','schedule_week2']
    Xtest = pd.concat([Xtest,temp_df], axis = 1)

    print('adding score/weather data')
    if test:
        Xtest['team_home_mod'] = Xtest.team_home_x.apply(lambda x: x.split(' ')[-1])
        Xtest['team_away_mod'] = Xtest.team_away_x.apply(lambda x: x.split(' ')[-1])
        Xtest['zipper_for_cols'] = list(zip(Xtest['team_home_mod'],
                                    Xtest['team_away_mod'],
                                    Xtest['schedule_week2']))
        #result_path = '/Users/Ben/Desktop/scripters/weather_score_2019_df.csv'
    
        result = score_weather

        result['zipper_for_cols'] = list(zip(result['home'],result['away'],result['week']))

        result_summ = pd.merge(Xtest,result,how = 'left',on = 'zipper_for_cols')
        current_week_df = result_summ[result_summ['week'] == curr_week]

        result_summ = result_summ[result_summ['week'] < curr_week]
        result_summ['score'] = result_summ['score'].str.replace('Final: ','').str.replace(' - ',',')
        result_summ['score'] = result_summ['score'].apply(lambda x: x.split(',')).apply(lambda x: list(reversed(x)))
        

        def func_binary(row):
            row = str(row)
            
            row = row.replace('[','').replace(']','').replace("'",'').replace(' ','')
            comma = row.index(',')
            diff = int(row[:comma]) - int(row[comma + 1:])
            
            if diff > 0:
                value = 1
            else:
                value = 0
            return value

        def func_continuous(row):
            row = str(row)
            
            row = row.replace('[','').replace(']','').replace("'",'').replace(' ','')
            comma = row.index(',')
            diff = int(row[:comma]) - int(row[comma + 1:])
            return diff

        result_summ['hometeam_win'] = result_summ['score'].apply(func_binary)
        result_summ['score_differential'] = result_summ['score'].apply(func_continuous)
        file_outpath = '/Users/Ben/Desktop/pwd_test/'
        result_summ.to_csv(file_outpath + 'Xtest_2019_weeks.csv', index = False)
        current_week_df.to_csv(file_outpath + 'week9_for_predictions.csv')
        return result_summ,current_week_df




    file_outpath = '/Users/Ben/Documents/scripters/'
    #Xtester.to_csv(file_outpath + 'Xtest_2019_week8.csv', index = False)
    # print('file saved in: ', file_outpath)
    return Xtester




