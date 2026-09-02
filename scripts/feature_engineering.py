import pandas as pd

def add_basic_features(team_summary):
    # Goal difference already exists (safe check)
    if 'goal_difference' not in team_summary.columns:
        team_summary['goal_difference'] = (
            team_summary['total_goals_scored'] - team_summary['total_goals_conceded']
        )

    return team_summary


def add_efficiency_features(team_summary):
    # Average goals scored per match
    team_summary['avg_goals_scored'] = (
        team_summary['total_goals_scored'] / team_summary['matches_played']
    )

    # Average goals conceded per match
    team_summary['avg_goals_conceded'] = (
        team_summary['total_goals_conceded'] / team_summary['matches_played']
    )

    return team_summary


def add_points_features(team_summary):
    # Points system (Win = 3, Draw = 1)
    team_summary['points'] = (
        team_summary['win'] * 3 + team_summary['draw']
    )

    # Points per match
    team_summary['points_per_match'] = (
        team_summary['points'] / team_summary['matches_played']
    )

    return team_summary


def add_all_features(team_summary):
    team_summary = add_basic_features(team_summary)
    team_summary = add_efficiency_features(team_summary)
    team_summary = add_points_features(team_summary)

    return team_summary
