def objective(space):

    eval_set  = [(xtrain,ytrain), (xvalid.values,yvalid)]
    xgb = XGBClassifier(n_estimators = 10000,
                        reg_lambda = space['reg_lambda'],
                        reg_alpha = space['reg_alpha'],
                        max_depth = space['max_depth'],
                        min_child_weight = space['min_child_weight'],
                        subsample = space['subsample'])
    
    xgb.fit(xtrain,ytrain,eval_set = eval_set,eval_metric="auc", early_stopping_rounds=30)

    
    pred = xgb.predict_proba(xvalid.values)[:,1]
    auc = roc_auc_score(yvalid, pred)
    print("SCORE:", auc)

    return{'loss':1-auc, 'status': STATUS_OK }


def get_weekly_preds(m):
    xgb_cols2 = ['current_year_average-scoring-margin_DIFF', 'streak_diff', 'away_opponent-extra-point-attempts-per-game_DIFF', 
    'away_opponent-extra-points-made-per-game_DIFF', 'home_offensive-points-per-game_DIFF', 
             'home_touchdowns-per-game_DIFF']
    log_cols2 = ['away_third-down-conversion-pct_DIFF', 'streak_diff', 'away_opponent-1st-half-points-per-game_DIFF']
    
    if m == 'xgb1':
        model = XGBClassifier(max_depth = 3, subsample = 0.5, min_child_weight = 5)
        cols = xgb_cols2
    if m == 'xgb3':
        model = XGBClassifier(max_depth = 3, subsample = .5, gamma = 2)
        cols = xgb_cols2
    if m == 'log':
        model = LogisticRegression(C = 1.0)
        cols = log_cols2
    if m == 'lda':
        model = LinearDiscriminantAnalysis()
        cols = log_cols2
    if m == 'gau':
        model = GaussianProcessClassifier()
        cols = log_cols2
    if m == 'svc':
        model = SVC(C=1.0)
        cols = log_cols2
        
    model.fit(xtrain[cols],ytrain)
    full_valid = pd.concat([xvalid,yvalid],axis=1)
    eteen_weekly_dict = {}
    nteen_weekly_dict = {}
    for year in xvalid.schedule_season.unique():
        full_yearly = full_valid[full_valid.schedule_season == year]
        for week in full_yearly.schedule_week.unique():
            Xer = full_yearly[full_yearly.schedule_week == week][cols]
            Yer = full_yearly[full_yearly.schedule_week == week]['spread_target']
            preds = model.predict(Xer)
            if year == 2018:
                eteen_weekly_dict[week] = accuracy_score(Yer,preds)
            if year == 2019:
                nteen_weekly_dict[week] = accuracy_score(Yer,preds)
    return eteen_weekly_dict,nteen_weekly_dict