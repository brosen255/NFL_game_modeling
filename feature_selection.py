def prepare_data(filepath):
	full,_,_ = process_df(fp=filepath)
	full['Money_home'] = full['Money_home'].fillna(full.Money_home.mean())
	full.drop([i for i in full.columns if i.startswith('year')],axis=1,inplace=True)
	full['offdef_eff'] = full['current_year_yards-per-point_DIFF'] - full['current_year_opp-yards-per-point_DIFF']

	eff_col = full['offdef_eff']
	full.drop(labels=['offdef_eff'], axis=1,inplace = True)
	full.insert(4, 'offdef_eff', eff_col)
	fulltest = process_df(fp='test_set_post_mr.csv',test = True)
	fulltest_orig = fulltest.copy()


	full = full[full.schedule_season >= 2011]
	return split_data(full)


def Feature_Selection_method1(xfull,yfull,n=5,prior_year_cols = True):
    if not prior_year_cols:
        print('Excluding last year cols')
        subset_diff_cols= [i for i in xfull.columns if 'last_year' not in i if i != 'game_info']
        subset_diff_cols= [i for i in subset_diff_cols if 'last_1_' not in i]
        xfull = xfull[subset_diff_cols]
    start_column = start_col(xfull)
    print('getting preliminary cols')
    prelim_list = prelim_selection(xfull,yfull,col_groups(xfull,start_column))
    add_cols=[ i for i in list(xfull.columns[:start_column]) if i != 'game_info']
    trunc_col_list = prelim_list + add_cols
    print('getting top {} cols'.format(n))
    predictors = top_vars(xfull,yfull,trunc_col_list,n)
    return predictors
method1_vars = FE_method1(xtrain,ytrain,n=5,prior_year_cols = True)

def method2_feat_score(xtrain,ytrain,keep_last_year = False):
    base_cols = ['current_year_average-scoring-margin_DIFF','away_third-down-conversion-pct_DIFF']
    base_cols = ['last_3_opponent-giveaway-fumble-recovery-pct_DIFF',
 'away_defensive-touchdowns-per-game_DIFF']
    ao_cols = [i for i in xtrain.columns if i not in base_cols if i != 'game_info']
    if not keep_last_year:
        ao_cols = [i for i in ao_cols if 'last_year' not in i if 'last_1' not in i]
    metric_to_use = 'accuracy'
    model = LogisticRegression(C = 1.0)
    f1_dict = {}
    print('scoring each feature')
    for col in ao_cols:
        lidx = ao_cols.index(col)
        if str(lidx)[-1] == '0':
            prog = round(lidx/len(ao_cols),2)
            if len(str(prog)) == 3:
                print(prog)
        cc = base_cols + [col]
        f1_dict[col] = cross_val_score(model, xtrain[cc],ytrain,scoring = 'accuracy',cv=3).mean()
    return f1_dict,base_cols

def top_feats(xtrain,ytrain,dict_,base_cols,n):    
    idx_list = find_top_indices(dict_.values(),n)
    top20 = [list(dict_.keys())[i] for i in idx_list]
    cols_to_test = base_cols+top20
    return cols_to_test
f1_dict,base_cols = method2_feat_score(xtrain,ytrain,keep_last_year = True)
cols_to_test = top_feats(xtrain,ytrain,f1_dict,base_cols,20)

def FE_method2(xtrain,ytrain,cols_,method_,metric_to_use = 'accuracy'):

    # Build step forward feature selection
    model = LogisticRegression()
    #model = SVC(C = 0.4)
    print('forward feature selection')
    sfs1 = sfs(model,
               k_features=method_,
               forward=True,
               floating=False,
               verbose=0,
               scoring=metric_to_use,
               cv=5)

    
    # Perform SFFS
    sfs1 = sfs1.fit(xtrain[cols_], ytrain)
    selected = list(sfs1.k_feature_names_)
    return selected
