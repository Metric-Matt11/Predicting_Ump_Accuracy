import pandas as pd
import matplotlib.pyplot as plt
from pybaseball import schedule_and_record
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

""" 
Umpire accuracy data comes from the umpscorecards.com datasets
Clean W/L columns. Games up should be + and down should be -
"""

parks = pd.read_csv('datasets/Park_Mapping.csv')
parks_loc = parks[['Team Abv', 'Arena Location']]
games = pd.read_csv('datasets/umpgames.csv')
umpires = pd.read_csv('datasets/umpstats.csv')
time_record_by_game = pd.DataFrame()

games['Date_team'] = games['Date'] + games['Home Team']
parks_list = parks['Pybaseball Abv'].tolist()

# Code snipet to clean date col and put in other tables format
for teams in parks_list:
    time_and_record = schedule_and_record(2021, teams)
    time_record_by_game = time_record_by_game.append(time_and_record)

time_record_by_game['Doubleheader Game'] = time_record_by_game['Date'].apply(
    lambda dh: dh[dh.find("(") + 1:dh.find(")")] if dh.find(')') > 0 else 0)
time_record_by_game['Date'] = time_record_by_game['Date'].map(lambda date: date[:-4] if date.find(')') > 0 else date)
time_record_by_game['Date_team'] = time_record_by_game['Date'] + time_record_by_game['Tm']
time_record_by_game = time_record_by_game[(time_record_by_game['D/N'] == 'D') | (time_record_by_game['D/N'] == 'N')]
time_record_by_game['GB'] = time_record_by_game['GB'].replace(['Tied'], 0)
time_record_by_game['GB'] = time_record_by_game['GB'].str.replace('up ', '-')
time_record_by_game['GB'] = time_record_by_game['GB'].astype(float) * -1
time_record_by_game['GB'] = time_record_by_game['GB'].fillna(0)

#Creating index of date, home team and Opp so I can match away and home team data for GB column



umpire_games = pd.merge(games, umpires, how='inner', on='Umpire', suffixes=("", "_season"))
umpire_games_parks = pd.merge(umpire_games, parks_loc, how='inner', left_on='Home Team', right_on='Team Abv')
umpire_games_parks_time = pd.merge(umpire_games_parks, time_record_by_game, how='inner', on='Date_team')
umpire_games_parks_time = umpire_games_parks_time.drop_duplicates(subset=['ID'])

usable_cols = ['Date_x', 'Umpire', 'Home Team', 'Away Team', 'Home Score', 'Away Score', 'Accuracy',
               'Consistency', 'Favor [Home]', 'Games', 'Called Pitches_season', 'Accuracy_season',
               'Arena Location', 'W/L', 'R', 'Inn', 'GB', 'Time', 'D/N', 'Attendance', 'Streak',
               'Doubleheader Game']

usable_data = umpire_games_parks_time[usable_cols]

model_cols = ['Games', 'Called Pitches_season', 'Accuracy_season',
              'Inn', 'GB', 'Attendance', 'Streak',
              'Doubleheader Game']

model_data = umpire_games_parks_time[model_cols]
model_data = model_data.fillna(0)


def plot_scatter(x_axis, y_axis, input_data):
    plt.scatter(x_axis, y_axis, data=input_data)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.show()


#def rand_forest_model(x_data, predi)
x = model_data
y = usable_data.Accuracy
train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=1)
performance_variables = []
umpire_performance_model = RandomForestRegressor(random_state=1)
umpire_performance_model.fit(train_x, train_y)
umpire_predictions = umpire_performance_model.predict(val_x)
print(umpire_predictions)
print(mean_absolute_error(val_y, umpire_predictions))
