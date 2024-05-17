import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
import seaborn as sns

# Load and preprocess data
Matches = pd.read_csv('data/WorldCupMatchesData.csv')
Champion = pd.read_csv("data/WorldCupsData.csv")
Ranking = pd.read_csv("data/fifa-ranking-04-04-2024.csv", parse_dates=["rank_date"])

start_date = pd.to_datetime('2023-01-01')
end_date = pd.to_datetime('2024-04-05')

Ranking = Ranking[(Ranking["rank_date"] > start_date) & (Ranking["rank_date"] < end_date)]
Ranking = Ranking.drop(["rank", "country_abrv", "previous_points", "confederation", "rank_change", "rank_date"], axis=1)
Champion = Champion.drop(["Year", "Country", "Runners-Up", "Third", "Fourth", "GoalsScored", "QualifiedTeams", "MatchesPlayed"], axis=1)
matches = Matches.drop(["Year"], axis=1)


matches = pd.merge(matches, Ranking, left_on="Home Team Name", right_on="country_full", how="left")
matches = pd.merge(matches, Ranking, left_on="Away Team Name", right_on="country_full", how="left", suffixes=("_home", "_away"))
champion_count = Champion["Winner"].value_counts().to_dict()

# Map team name to numbers
def indexing_theteams(df):
    teams = {}
    index = 0
    for lab, row in matches.iterrows():
        if row["Home Team Name"] not in teams.keys():
            teams[row["Home Team Name"]] = index
            index += 1
        if row["Away Team Name"] not in teams.keys():
            teams[row["Away Team Name"]] = index
            index += 1
    return teams

teams_index = indexing_theteams(matches)

matches["Championship Home"] = 0
matches["Championship Away"] = 0

def get_champion(row):
    if row["Home Team Name"] in champion_count:
        row["Championship Home"] = champion_count[row["Home Team Name"]]
    if row["Away Team Name"] in champion_count:
        row["Championship Away"] = champion_count[row["Away Team Name"]]
    return row

matches = matches.apply(get_champion, axis=1)

matches["Home Team Name"] = matches["Home Team Name"].apply(lambda x: teams_index[x])
matches["Away Team Name"] = matches["Away Team Name"].apply(lambda x: teams_index[x])



matches["Who Wins"] = 0
matches["Goal Difference"] = matches["Home Team Goals"] - matches["Away Team Goals"]

def gettingwhowins(df):
    if df["Goal Difference"] == 0:
        df["Who Wins"] = 0  # Draw
    elif df["Goal Difference"] > 0:
        df["Who Wins"] = 1  # Home team wins
    else:
        df["Who Wins"] = 2  # Away team wins
    return df

matches = matches.apply(gettingwhowins, axis=1)
matches = matches.drop(["country_full_home", "country_full_away"], axis=1)

matches = matches.dropna()

matches = matches.drop(["Home Team Goals", "Away Team Goals", "Goal Difference"], axis=1)

print("matches:\n",matches)
matches.to_csv('./merged.csv')

# Splitting Data
X = matches[["Home Team Name", "Away Team Name", "total_points_home", "total_points_away", "Championship Home", "Championship Away"]].values
y = matches["Who Wins"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
print("Train data: ", X_train)

# Model Training
rf = RandomForestClassifier(random_state=42)

# Validate model
param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20]}
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Model Evaluation
best_rf = grid_search.best_estimator_
y_pred_rf = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy of Random Forest: {accuracy* 100:.3f}%")
print("Best Parameters:", grid_search.best_params_)

# Feature Importance
feature_importances = best_rf.feature_importances_
feature_names = ["Home Team Name", "Away Team Name", "Total Points Home", "Total Points Away"]

print("Random Forest Classifier Feature Importances:")
for name, importance in zip(feature_names, feature_importances):
    print(f"{name}: {importance* 100:.3f}%")

models = {
    'RandomForest': RandomForestClassifier(random_state=42, n_estimators=200, max_depth=10),
    'LogisticRegression': LogisticRegression(max_iter=10000, solver='saga')  # Using 'saga' solver for L1 and ElasticNet regularization support
}

# Define parameter grid for logistic regression
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100]  # Regularization strength
}

# Perform grid search for logistic regression
grid_search_lr = GridSearchCV(LogisticRegression(max_iter=10000, solver='saga'), param_grid_lr, cv=5, scoring='accuracy')
grid_search_lr.fit(X_train, y_train)

# Get the best logistic regression model
best_lr = grid_search_lr.best_estimator_

# Evaluate the best logistic regression model
y_pred_lr = best_lr.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Accuracy of the best Logistic Regression: {accuracy_lr * 100:.3f}%")
print("Best Parameters for Logistic Regression:", grid_search_lr.best_params_)



results = {}

# RandomForestClassifier
best_rf.fit(X_train, y_train)
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
results['RandomForestClassifier'] = accuracy

# Logistic Regression
best_lr.fit(X_train, y_train)
y_pred = best_lr.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
results['LogisticRegression'] = accuracy

# Update best lr model
models['LogisticRegression'] = best_lr
models["RandomForest"] = best_rf
# Comparing Results
print("\nModel Accuracy Comparison:")
for name, accuracy in results.items():
    print(f"{name}: {accuracy * 100:.3f}%")

# Confusion Matrix & Classification Report
# Random Forest Classifier
cm_rf = confusion_matrix(y_test, y_pred_rf)
print("Confusion Matrix RF:\n", cm_rf)
print("Classification Report:\n", classification_report(y_test, y_pred_rf, zero_division=1))
# Logistic Regression
cm_lr = confusion_matrix(y_test, y_pred_lr)
print("Confusion Matrix LR:\n", cm_lr)
print("Classification Report for Logistic Regression:\n", classification_report(y_test, y_pred_lr, zero_division=1))

# Visualize the 2 Confusion Matrices
# Random Forest Classifier
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Random Forest Classifier")
plt.show()
# Logistic Regression
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="YlOrBr")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Logistic Regression")
plt.show()

# Print Training Data
print("Training Data:")
print(matches.head())

# Detailed Single Match Prediction
def detailed_single_match_prediction(model, team1, team2):
    x = []
    
    try:
        x.append(teams_index[team1])
        x.append(teams_index[team2])
        x.append(Ranking.loc[Ranking["country_full"] == team1, "total_points"].values[0])
        x.append(Ranking.loc[Ranking["country_full"] == team2, "total_points"].values[0])
        x.append(champion_count.get(team1, 0))
        x.append(champion_count.get(team2, 0))
        x = np.array(x).reshape(1, -1)
        
        # print("x: ", x)
        print('-----------------------------------------------------------------------------')
        print(f"\nModel used: {model}")
        print("Input Features for Prediction:")
        print(f"Home - Away: {team1} - {team2}")
        print(f"Total Points Home: {x[0][2]}")
        print(f"Total Points Away: {x[0][3]}")
        print(f"Championship Home: {x[0][4]}")
        print(f"Championship Away: {x[0][5]}")
        print('-----------------------------------------------------------------------------')

        prediction = model.predict(x)[0]
        prediction_proba = model.predict_proba(x)[0]
        
        print("Prediction and Probabilities:")
        print(f"Predicted Winner: {'Home Team' if prediction == 1 else 'Away Team' if prediction == 2 else 'Draw'}")
        print(f"Draw Probability: {prediction_proba[0]* 100:.3f}%")
        print(f"Home Team Win Probability: {prediction_proba[1]* 100:.3f}%")
        print(f"Away Team Win Probability: {prediction_proba[2]* 100:.3f}%")
        
        return prediction, prediction_proba
    except Exception as e:
        print("Not valid names")
        raise e

# Example single match prediction
print('Example single match prediction')
team1 = "Iran"
team2 = "Portugal"
result, probabilities = detailed_single_match_prediction(best_rf, team1, team2)

# Example single match prediction with logistic regression
result_lr, probabilities_lr = detailed_single_match_prediction(best_lr, team1, team2)


def getting_input(model, team1, team2, is_logistic_regression=False):
    if team1 == team2:
        print("Invalid names")
        raise ValueError("Same Names")
    x = []
    
    try:
        x.append(teams_index[team1])
        x.append(teams_index[team2])
        x.append(Ranking.loc[Ranking["country_full"] == team1, "total_points"].values[0])
        x.append(Ranking.loc[Ranking["country_full"] == team2, "total_points"].values[0])        
        x.append(champion_count.get(team1, 0))
        x.append(champion_count.get(team2, 0))
        
        x = np.array(x).reshape(1, -1)
    except Exception as e:
        print("Not valid names")
        raise e

    return model.predict(x)[0], model.predict_proba(x)[0]

def determine_winner(result, prob, match):
    if result == 1:
        return match[0]
    elif result == 2:
        return match[1]
    elif prob[1] > prob[2]:
        return match[0]
    else:
        return match[1]

def mua_giai(arr, model, is_logistic_regression=False):
    if len(arr) == 1:
        result, prob = getting_input(model, arr[0][0], arr[0][1])
        return determine_winner(result, prob, arr[0])
    
    if len(arr) % 2 == 1:
        raise ValueError("Not valid team")

    next_round = []
    for i in range(0, len(arr), 2):
        match1 = arr[i]
        match2 = arr[i + 1]

        result1, prob1 = detailed_single_match_prediction(model, match1[0], match1[1])
        result2, prob2 = detailed_single_match_prediction(model, match2[0], match2[1])

        winner1 = determine_winner(result1, prob1, match1)
        winner2 = determine_winner(result2, prob2, match2)

        next_round.append([winner1, winner2])

    return mua_giai(next_round, model)

matches = [
    ["Croatia", "Brazil"],
    ["Netherlands", "Argentina"],
    ["Morocco", "Portugal"],
    ["England", "France"]
]

result = mua_giai(matches, models["RandomForest"])
print(result)

result = mua_giai(matches, models["LogisticRegression"])
print(result)