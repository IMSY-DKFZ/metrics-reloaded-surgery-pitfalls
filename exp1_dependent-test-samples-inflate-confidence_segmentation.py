import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm


def bootstrap_hierarchique_one_level(df, n_iterations, score):
    """
    Hierarchical bootstrap on one level: surgery

    Parameters:
    - df : DataFrame pandas with columns 'surgery_number' and score.
    - n_iterations : number of bootstrap samples.
    - score: metric (DSC or NSD)

    Returns:
    - bootstrap_samples : list of boostrap means
    """

    bootstrap_samples = []

    surgery_numbers = df['surgery_number'].unique()

    for _ in range(n_iterations):

        bootstrap_data = []

        sampled_surgery_numbers = np.random.choice(surgery_numbers, size=len(surgery_numbers), replace=True)

        for surgery_number in sampled_surgery_numbers:
            surgery_data = df[df['surgery_number'] == surgery_number]

            sampled_dsc = np.random.choice(surgery_data[score].values, size=len(surgery_data[score]), replace=True)

            bootstrap_data.extend(sampled_dsc)

        bootstrap_sample = pd.DataFrame({score: bootstrap_data})

        bootstrap_samples.append(np.mean(bootstrap_sample))

    return bootstrap_samples

# TODO add data path
# TODO file name: robustmis_results_concat_binary_segmentation.csv
file_path = ''

try:
  df = pd.read_csv(file_path)
  print("File loaded successfully.")
  # Display the first few rows of the DataFrame (optional)
  print(df.head())
except FileNotFoundError:
  print(f"Error: File not found at {file_path}. Please check the file path.")
except pd.errors.ParserError:
  print(f"Error: Could not parse the file at {file_path}. Please check the file format.")
except Exception as e:
  print(f"An unexpected error occurred: {e}")

df = df.dropna(subset=['DSC'])
# If NSD should be used, replace DSC by NSD

# create a new df with all lines for which stage=Stage_3

df_stage_3 = df[df['stage'] == 'Stage_3']
print(df_stage_3)

team_map = {
    "haoyun": "A1",
    "CASIA_SRL": "A2",
    "www": "A3",
    "fisensee": "A4",
    "Uniandes": "A5",
    "SQUASH": "A6",
    "Caresyntax": "A7",
    "Djh": "A8",
    "VIE": "A9",
    "NCT": "A10"
}

# replace names in the 'team_name' column
df_stage_3["team_name"] = df_stage_3["team_name"].map(team_map)

print(df_stage_3)

teams = df_stage_3['team_name'].unique()

results = []


def bootstrap_ci(bootstrap_results, confidence=0.95):
    lower = np.percentile(bootstrap_results, (1 - confidence) / 2 * 100)
    upper = np.percentile(bootstrap_results, (1 + confidence) / 2 * 100)
    return lower, upper


def naive_bootstrap(data, nboot=1000):
    bootstrap_means = []
    for _ in range(nboot):
        resampled_data = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(resampled_data))
    return bootstrap_means


for i, team in enumerate(teams):
    print(f"Processing team: {team}")

    team_data = df_stage_3[df_stage_3['team_name'] == team]

    dsc_values = team_data['DSC'].dropna().values

    bootstrap_naive_results = naive_bootstrap(dsc_values, nboot=1000)

    ci_naive_bootstrap = bootstrap_ci(bootstrap_naive_results)

    bootstrap_results = bootstrap_hierarchique_one_level(team_data, 1000, 'DSC')

    results.append({'team': team, 'type': 'bootstrap_naive',
                    'ci_lower': ci_naive_bootstrap[0], 'ci_upper': ci_naive_bootstrap[1]})
    results.append({'team': team, 'type': 'bootstrap_hierarchical',
                    'ci_lower': bootstrap_ci(bootstrap_results)[0], 'ci_upper': bootstrap_ci(bootstrap_results)[1]})

results_df = pd.DataFrame(results)

print(results_df)

plt.figure(figsize=(12, 8))

colors = {
    'bootstrap_naive': 'tab:blue',
    'bootstrap_hierarchical': 'tab:orange',
}

legend_labels = []

team_order = [f"A{i}" for i in range(1, 11)]
results_df["team"] = pd.Categorical(results_df["team"], categories=team_order, ordered=True)
results_df = results_df.sort_values("team")

for i, team in enumerate(results_df['team'].unique()):
    team_results = results_df[results_df['team'] == team]
    for j, row in enumerate(team_results.itertuples()):
        x_position = i + j * 0.1

        plt.plot([x_position, x_position], [row.ci_lower, row.ci_upper],
                 marker="_", color=colors[row.type], linewidth=1.5)

        if row.type not in legend_labels:
            plt.plot([], [], color=colors[row.type], label=row.type)
            legend_labels.append(row.type)

plt.title('Confidence Interval bootstrap naive vs hierarchical')
plt.xticks(range(len(results_df['team'].unique())), results_df['team'].unique(), rotation=90)
plt.xlabel('Team')
plt.ylabel('Confidence Intervals')
plt.legend(title='Methods', loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()

plt.show()

# calculate CI width
results_df["ci_width"] = results_df["ci_upper"] - results_df["ci_lower"]

print(results_df)

# pivot the dataframe so that both CI widths are columns
pivot_df = results_df.pivot(index="team", columns="type", values="ci_width")

# calculate ratio hierarchical / naive
pivot_df["ci_ratio"] = pivot_df["bootstrap_hierarchical"] / pivot_df["bootstrap_naive"]

print(pivot_df)

print("\nMedian ratio:")
print(np.median(pivot_df["ci_ratio"]))