import pandas as pd

def prepare_team_data(df):
    # Home data
    home = df[['home_team', 'home_goals', 'away_goals']].copy()
    home.columns = ['team', 'goals_scored', 'goals_conceded']

    # Away data
    away = df[['away_team', 'away_goals', 'home_goals']].copy()
    away.columns = ['team', 'goals_scored', 'goals_conceded']

    return pd.concat([home, away])


def compute_team_summary(team_data):
    summary = team_data.groupby('team').agg(
        matches_played=('team', 'count'),
        total_goals_scored=('goals_scored', 'sum'),
        total_goals_conceded=('goals_conceded', 'sum')
    )

    summary['goal_difference'] = (
        summary['total_goals_scored'] - summary['total_goals_conceded']
    )

    return summary.reset_index()


def add_match_results(df):
    # Home results
    df['home_result'] = df.apply(
        lambda row: 'win' if row['home_goals'] > row['away_goals']
        else ('loss' if row['home_goals'] < row['away_goals'] else 'draw'),
        axis=1
    )

    # Away results
    df['away_result'] = df.apply(
        lambda row: 'win' if row['away_goals'] > row['home_goals']
        else ('loss' if row['away_goals'] < row['home_goals'] else 'draw'),
        axis=1
    )

    return df


def compute_results_summary(df):
    # Home results
    home = df[['home_team', 'home_result']].copy()
    home.columns = ['team', 'result']

    # Away results
    away = df[['away_team', 'away_result']].copy()
    away.columns = ['team', 'result']

    results = pd.concat([home, away])

    return results.groupby(['team', 'result']).size().unstack(fill_value=0)