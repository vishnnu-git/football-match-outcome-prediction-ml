import pandas as pd

def load_data():
    # Load datasets
    df = pd.read_csv("data/matches_clean.csv")
    teams = pd.read_csv("data/Team.csv")

    # Keep only required columns
    teams = teams[['team_api_id', 'team_long_name']]

    return df, teams