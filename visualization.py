import matplotlib.pyplot as plt

def plot_goal_difference(df):
    top10 = df.head(10)

    plt.figure()
    plt.bar(top10['team_long_name'], top10['goal_difference'])
    plt.title("Top 10 Teams by Goal Difference")
    plt.xlabel("Team")
    plt.ylabel("Goal Difference")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig("outputs/top10_goal_difference.png")
    plt.show()


def plot_goals_scored(df):
    top10 = df.sort_values(by='total_goals_scored', ascending=False).head(10)

    plt.figure()
    plt.bar(top10['team_long_name'], top10['total_goals_scored'])
    plt.title("Top 10 Teams by Goals Scored")
    plt.xlabel("Team")
    plt.ylabel("Goals Scored")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig("outputs/top10_goals_scored.png")
    plt.show()


def plot_goals_conceded(df):
    top10 = df.sort_values(by='total_goals_conceded', ascending=False).head(10)

    plt.figure()
    plt.bar(top10['team_long_name'], top10['total_goals_conceded'])
    plt.title("Top 10 Teams by Goals Conceded")
    plt.xlabel("Team")
    plt.ylabel("Goals Conceded")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig("outputs/top10_goals_conceded.png")
    plt.show()


def plot_win_rate(df):
    top10 = df.sort_values(by='win_rate', ascending=False).head(10)

    plt.figure()
    plt.bar(top10['team_long_name'], top10['win_rate'])
    plt.title("Top 10 Teams by Win Rate")
    plt.xlabel("Team")
    plt.ylabel("Win Rate")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig("outputs/top10_win_rate.png")
    plt.show()