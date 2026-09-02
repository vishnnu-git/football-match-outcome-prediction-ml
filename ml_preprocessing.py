import pandas as pd
import os
from sklearn.model_selection import train_test_split


def preprocess():
    print("⚙️ Creating ML dataset...")

    # Load cleaned match data
    df = pd.read_csv("data/matches_clean.csv")
    df = df.sort_values("date")

    features = []
    team_stats = {}

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        # Initialize teams
        for team in [home, away]:
            if team not in team_stats:
                team_stats[team] = {
                    "goals_scored": 0,
                    "goals_conceded": 0,
                    "matches": 0,
                    "wins": 0,
                    "recent_results": []
                }

        home_stats = team_stats[home]
        away_stats = team_stats[away]

        # Skip early matches
        if home_stats["matches"] < 5 or away_stats["matches"] < 5:
            home_stats["matches"] += 1
            away_stats["matches"] += 1

            home_stats["goals_scored"] += row["home_goals"]
            home_stats["goals_conceded"] += row["away_goals"]

            away_stats["goals_scored"] += row["away_goals"]
            away_stats["goals_conceded"] += row["home_goals"]
            continue

        # Feature calculations
        home_avg_goals = home_stats["goals_scored"] / home_stats["matches"]
        away_avg_goals = away_stats["goals_scored"] / away_stats["matches"]

        home_win_rate = home_stats["wins"] / home_stats["matches"]
        away_win_rate = away_stats["wins"] / away_stats["matches"]

        home_recent = sum(home_stats["recent_results"][-5:]) / 5 if len(home_stats["recent_results"]) >= 5 else 0
        away_recent = sum(away_stats["recent_results"][-5:]) / 5 if len(away_stats["recent_results"]) >= 5 else 0

        # Features
        goal_diff_strength = home_avg_goals - away_avg_goals
        win_rate_diff = home_win_rate - away_win_rate
        defense_diff = (away_stats["goals_conceded"] / away_stats["matches"]) - \
                       (home_stats["goals_conceded"] / home_stats["matches"])
        form_diff = home_recent - away_recent

        # Target
        result = 1 if row["home_goals"] > row["away_goals"] else 0

        features.append({
            "home_avg_goals": home_avg_goals,
            "away_avg_goals": away_avg_goals,
            "home_win_rate": home_win_rate,
            "away_win_rate": away_win_rate,
            "goal_diff_strength": goal_diff_strength,
            "win_rate_diff": win_rate_diff,
            "defense_diff": defense_diff,
            "home_recent_form": home_recent,
            "away_recent_form": away_recent,
            "form_diff": form_diff,
            "result": result
        })

        # Update stats
        home_stats["matches"] += 1
        away_stats["matches"] += 1

        home_stats["goals_scored"] += row["home_goals"]
        home_stats["goals_conceded"] += row["away_goals"]

        away_stats["goals_scored"] += row["away_goals"]
        away_stats["goals_conceded"] += row["home_goals"]

        if row["home_goals"] > row["away_goals"]:
            home_stats["wins"] += 1
            home_stats["recent_results"].append(1)
            away_stats["recent_results"].append(0)
        else:
            away_stats["wins"] += 1
            home_stats["recent_results"].append(0)
            away_stats["recent_results"].append(1)

    # Create DataFrame
    ml_df = pd.DataFrame(features)

    # Save
    os.makedirs("outputs", exist_ok=True)
    ml_df.to_csv("outputs/ml_dataset.csv", index=False)

    print("✅ ML dataset created successfully!")
    print(f"Rows: {len(ml_df)}")

    # ===============================
    # SPLIT DATA
    # ===============================
    X = ml_df.drop("result", axis=1)
    y = ml_df["result"]

    return train_test_split(X, y, test_size=0.2, random_state=42)