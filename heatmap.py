import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mpath


def draw_nhl_rink_with_clean_heatmap(df, title='Shot Heatmap', cmap='jet_r'):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor('white')

    # Plot heatmap
    sns.kdeplot(
        data=df, x='xCordAdjusted', y='yCordAdjusted',
        fill=True, cmap=cmap, bw_adjust=1,
        levels=100, thresh=0.05,
        alpha=0.6, ax=ax, zorder=0,  # lower z-order
        warn_singular=False
    )

    # Rink graphics
    def add_patch(patch):
        ax.add_patch(patch)

    # Rink boundary
    add_patch(patches.FancyBboxPatch(
        (-100, -42.5), 200, 85,
        boxstyle=patches.BoxStyle("Round", pad=0, rounding_size=28),
        linewidth=2, edgecolor='black', facecolor='None', zorder=2
    ))

    # Lines
    ax.plot([0, 0], [-42.5, 42.5], color='red', linewidth=2, zorder=2)
    ax.plot([25, 25], [-42.5, 42.5], color='blue', linewidth=5, zorder=2)
    ax.plot([89, 89], [-38.5, 38.5], color='red', linewidth=2, zorder=2)

    # Creases
    for x in [89]:
        add_patch(patches.Arc((x, 0), 12, 12, theta1=90, theta2=270, color='blue', linewidth=2, zorder=2))
    
    # Faceoff circles
    for x, y in [(69, 22), (69, -22)]:
        add_patch(patches.Circle((x, y), 15, fill=False, color='red', linewidth=2, zorder=2))
        add_patch(patches.Circle((x, y), 0.5, color='red', zorder=2))

    # Neutral zone dots
    for x, y in [(20, 22), (20, -22)]:
        add_patch(patches.Circle((x, y), 0.5, color='red', zorder=2))

    # Center circle
    add_patch(patches.Arc((0, 0), 15, 15, theta1=270, theta2=90, color='blue', linewidth=2, zorder=2))
    add_patch(patches.Arc((0, 0), 1, 1, theta1=270, theta2=90, color='blue', linewidth=4, zorder=2))

    ax.set_xlim(0, 100)
    ax.set_ylim(-42.5, 42.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=16)

    plt.tight_layout()
    plt.show()



df = pd.read_csv("shots_2024.csv")

player_name = input("Enter player name: ")
filtered_df = df[
    (df['shooterName'] == player_name) &
    (df['homeSkatersOnIce'] == df['awaySkatersOnIce']) &
    (df['isPlayoffGame'] == 0) 
]


draw_nhl_rink_with_clean_heatmap(filtered_df, title=f"{player_name} Shot Heatmap")