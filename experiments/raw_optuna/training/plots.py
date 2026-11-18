"""Plot measured model comparisons and Bayesian search results."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    directory = Path(__file__).resolve().parents[1] / 'results'
    report = json.loads((directory / 'experiments.json').read_text())
    rows = report['results']
    names = [row['model'].replace('_', ' ') +
             (' (tuned)' if row['variant'] == 'tuned' else '') for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout='constrained')
    metrics = [('cv_macro_f1', 'Training CV Macro F1'),
               ('top1_accuracy', 'Student-test Top-1 accuracy'),
               ('macro_f1', 'Student-test Macro F1'),
               ('log_loss', 'Student-test log loss (lower is better)')]
    for ax, (key, title) in zip(axes.flat, metrics):
        y = np.arange(len(rows))
        ax.barh(y, [row[key] for row in rows], color='#307873')
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.grid(axis='x', alpha=.25)
        ax.set_axisbelow(True)
    fig.savefig(directory / 'model_comparison.png', dpi=150)
    plt.close(fig)

    trials = report['trials']
    x = [trial['trial'] for trial in trials]
    scores = [trial['mean_cv_macro_f1'] for trial in trials]
    fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')
    ax.scatter(x, scores, label='Trial score')
    ax.step(x, np.maximum.accumulate(scores), where='post', label='Best so far')
    ax.set(xlabel='Trial', ylabel='Training CV Macro F1', xticks=x)
    ax.legend()
    fig.savefig(directory / 'bayesian_search.png', dpi=150)


if __name__ == '__main__':
    main()
