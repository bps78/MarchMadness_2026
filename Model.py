import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from colorama import Fore
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("prepped_data.csv")
matchups = pd.read_csv("2026_prepped_Matchups.csv")

features = ['T1_OffRating', 'T2_OffRating', 'T1_DefRating', 'T2_DefRating', 'Seed_Diff', 'offRatingDiff',
            'defRatingDiff', 'ftRateDiff', 'wabDiff', 'talentDiff', 'sosDiff', 'threepgDiff', 'tempoDiff', 'effHeightDiff',
             'defRating_x_tempo', 'talent_x_sos', 'T1_var', 'T2_var']

train = data[data['Season'] < 2020]
test = data[data['Season'] > 2020]

Xtrain = train[features]
ytrain = (train['T1_Score'] > train['T2_Score']).astype(int)
Xtest = test[features]
ytest = (test['T1_Score'] > test['T2_Score']).astype(int)

#Random Forest for Feature Selection
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(Xtrain, ytrain)

feature_importance = pd.Series(rf.feature_importances_, index = Xtrain.columns)
n = len(features)
top_features = feature_importance.nlargest(n).index #Selects the n top features
print(top_features)

#Filter for only selected features
Xtrain_selected = Xtrain[top_features]
Xtest_selected = Xtest[top_features]

#Train the XGBoost model
m1 = XGBClassifier()
m1.fit(Xtrain_selected, ytrain)
predictions = m1.predict_proba(Xtest_selected)

output = pd.DataFrame(predictions[:,1], columns = ['Predictions'])
output['Actual'] = ytest.astype(int).reset_index(drop=True)
output["Score"] = (output["Actual"] - output["Predictions"])**2

# Score = 0.25 indicates each team is given an equal chance, no pattern association
print(Fore.BLUE + "********** PREDITCTION SCORE: " + Fore.YELLOW + str(output["Score"].mean()) + Fore.BLUE + " ***************" + Fore.WHITE)

#Optimized parameters to avoid overfitting and improve the model
param_grid = {
    'max_depth': [5, 7, 10],
    'min_child_weight': [1, 3, 5],
    'learning_rate': [0.1, 0.01],
    'subsample': [0.7, 0.9, 1.0],
    'gamma': [0.1, 0.2, 0.5],
    'objective':["binary:logistic"],
    'eval_metric':["logloss"]
}

grid_search = GridSearchCV(estimator=m1, param_grid=param_grid, scoring='d2_log_loss_score', cv=5)
grid_search.fit(Xtrain_selected, ytrain)
best_params = grid_search.best_params_

#best_params = {'eval_metric': 'logloss', 'gamma': 0.1, 'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 3, 'objective': 'binary:logistic', 'subsample': 1.0}

print("Best Parameters: ", best_params)

m2 = XGBClassifier(**best_params)

m2.fit(Xtrain_selected, ytrain)
predictions2 = m2.predict_proba(Xtest_selected)

output = pd.DataFrame(predictions2[:,1], columns = ['Predictions'])
output['Actual'] = ytest.astype(int).reset_index(drop=True)
output["Score"] = (output["Actual"] - output["Predictions"])**2
print(Fore.BLUE + "********** OPTIMIZED PREDITCTION SCORE: " + Fore.GREEN + str(output["Score"].mean()) + Fore.BLUE + " ***************" + Fore.WHITE)
print(Fore.BLUE + "********** BEST PREDITCTION SCORE: " + Fore.RED + "0.1708748094011202" + Fore.BLUE + " ***************" + Fore.WHITE)

#Updated Predictions
test = test.copy()
test['Pred'] = predictions2[:,1]
test['rounded_preds'] = np.round(predictions2[:,1])
test['Actual'] = (test['T1_Score'] > test['T2_Score']).astype(int)
test['Correct'] = test['rounded_preds'] == test['Actual']
summary = test[['Season', 'DayNum', 'T1_TeamID', 'T1_Score', 'T2_TeamID', 'T2_Score', 'Pred', 'rounded_preds', 'Actual', 'Correct']]

#Save our Predictions for viewing
summary.to_csv('Predictions.csv', index=False)



#Make predictions for this year
finalPredictFeatures = matchups[top_features]
finalPredictions = m2.predict_proba(finalPredictFeatures)

finalPredictionsTable = matchups.copy()
matchups['preds'] = finalPredictions[:,1]
matchups['rounded_preds'] = np.round(finalPredictions[:,1])
finalPredSummary = matchups[['HigherSeed', 'HigherSeedNum', 'LowerSeed', 'LowerSeedNum', 'preds']]

finalPredSummary.to_csv("final_predictions.csv", index=False)

#See what upsets we are predicting to take place
finalPredUpsets = matchups.loc[matchups['preds'] < 0.5]
finalPredUpsets = finalPredUpsets[['HigherSeed', 'HigherSeedNum', 'LowerSeed', 'LowerSeedNum', 'preds']]
finalPredUpsets.to_csv("final_prediction_upsets.csv", index=False)

plt.hist(matchups['preds'], bins=20)
plt.savefig("prediction_distribution.png")