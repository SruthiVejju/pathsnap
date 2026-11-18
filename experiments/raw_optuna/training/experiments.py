"""Run five baselines and Optuna Bayesian tuning on PathSnap."""
import argparse
import csv
import io
import json
import platform
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import joblib
import numpy as np
import optuna
import scipy
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, log_loss, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from training.candidates import FEATURES, candidates, reference_labels

SEED = 42
K = 5


def read_trips(archive):
    splits = {'train': [], 'student_test': []}
    with ZipFile(archive) as source:
        for name in sorted(source.namelist()):
            parts = name.split('/')
            if len(parts) != 4 or parts[1] not in splits or not name.endswith('.track'):
                continue
            stem = name.removesuffix('.track')

            def load(extension, dtype=float, columns=2):
                return np.loadtxt(io.BytesIO(source.read(stem + extension)),
                                  dtype=dtype, ndmin=columns)

            track, nodes = load('.track'), load('.nodes')
            arcs, route = load('.arcs', int), load('.route', int, 1)
            ids, features = candidates(track, nodes, arcs, K)
            labels = reference_labels(track, nodes, arcs, route)
            splits[parts[1]].append((parts[2], ids, features, labels, arcs))
    return splits


def candidate_rows(trips):
    x = np.concatenate([t[2].reshape(-1, len(FEATURES)) for t in trips])
    y = np.concatenate([(t[1] == t[3][:, None]).ravel() for t in trips])
    return x, y


def fit(model, trips, limit):
    x, y = candidate_rows(trips)
    rng = np.random.default_rng(SEED)
    indices = rng.choice(len(y), min(limit, len(y)), replace=False)
    return model.fit(x[indices], y[indices])


def score(model, trips):
    x, y = candidate_rows(trips)
    start = perf_counter()
    probability = model.predict_proba(x)[:, 1]
    inference = perf_counter() - start
    classification = probability >= .5
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, classification, average='macro', zero_division=0)
    offset, hits, missed, points, connected, transitions = 0, 0, 0, 0, 0, 0
    for _, ids, _, labels, arcs in trips:
        end = offset + ids.size
        best = probability[offset:end].reshape(ids.shape).argmax(axis=1)
        predicted = ids[np.arange(len(ids)), best]
        hits += np.count_nonzero(predicted == labels)
        missed += np.count_nonzero(~(ids == labels[:, None]).any(axis=1))
        connected += np.count_nonzero((predicted[:-1] == predicted[1:]) |
                                      (arcs[predicted[:-1], 1] == arcs[predicted[1:], 0]))
        transitions += len(ids) - 1
        points += len(ids)
        offset = end
    return dict(macro_f1=f1, macro_precision=precision, macro_recall=recall,
                candidate_accuracy=float(np.mean(y == classification)),
                top1_accuracy=hits / points, candidate_miss_rate=missed / points,
                direct_route_continuity=connected / max(1, transitions),
                log_loss=log_loss(y, probability, labels=[False, True]),
                inference_seconds=inference,
                confusion_matrix=confusion_matrix(y, classification, labels=[False, True]).tolist())


FACTORIES = {
    'logistic': lambda p: make_pipeline(StandardScaler(), LogisticRegression(C=p[0], max_iter=300)),
    'gaussian_nb': lambda p: GaussianNB(var_smoothing=p[0]),
    'decision_tree': lambda p: DecisionTreeClassifier(max_depth=int(p[0]), min_samples_leaf=int(p[1]), random_state=SEED),
    'extra_trees': lambda p: ExtraTreesClassifier(n_estimators=32, max_depth=int(p[0]), min_samples_leaf=int(p[1]), random_state=SEED, n_jobs=2),
    'hist_gradient_boosting': lambda p: HistGradientBoostingClassifier(learning_rate=p[0], max_leaf_nodes=int(p[1]), max_iter=60, early_stopping=False, random_state=SEED),
}
DEFAULTS = {'logistic': [1.0], 'gaussian_nb': [1e-9], 'decision_tree': [12, 12],
            'extra_trees': [14, 12], 'hist_gradient_boosting': [.1, 31]}
SEARCH_SPACES = {
    'logistic': [('C', 1e-3, 100.0, True)],
    'gaussian_nb': [('var_smoothing', 1e-12, 1e-3, True)],
    'decision_tree': [('max_depth', 4, 24, False), ('min_samples_leaf', 4, 80, False)],
    'extra_trees': [('max_depth', 6, 26, False), ('min_samples_leaf', 4, 80, False)],
    'hist_gradient_boosting': [('learning_rate', .025, .25, False),
                               ('max_leaf_nodes', 8, 64, False)],
}


def cross_validate(factory, params, trips, folds, limit):
    return [score(fit(factory(params), [trips[i] for i in a], limit),
                  [trips[i] for i in b])['macro_f1'] for a, b in folds]


def bayesian_search(name, train, folds, limit, trials):
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=3),
    )

    def objective(trial):
        params = []
        for key, low, high, log_scale in SEARCH_SPACES[name]:
            if isinstance(low, int):
                params.append(trial.suggest_int(key, low, high))
            else:
                params.append(trial.suggest_float(key, low, high, log=log_scale))
        scores = cross_validate(FACTORIES[name], params, train, folds, limit)
        trial.set_user_attr('fold_macro_f1', scores)
        return float(np.mean(scores))

    study.optimize(objective, n_trials=trials)
    return [
        dict(trial=trial.number + 1, named_parameters=trial.params,
             parameters=[trial.params[key] for key, *_ in SEARCH_SPACES[name]],
             fold_macro_f1=trial.user_attrs['fold_macro_f1'],
             mean_cv_macro_f1=trial.value)
        for trial in study.trials
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    parser.add_argument('--fit-rows', type=int, default=60000)
    parser.add_argument('--trials', type=int, default=10, choices=range(4, 257), metavar='4..256')
    args = parser.parse_args()
    trips = read_trips(args.archive)
    train, test = trips['train'], trips['student_test']
    if len(train) != 80 or len(test) != 10:
        raise ValueError('Expected 80 training trips and 10 student-test trips.')
    ids = [trip[0] for trip in train]
    folds = list(GroupKFold(3).split(ids, groups=ids))
    results = []

    def evaluate(name, params, folds_scores, variant):
        start = perf_counter()
        model = fit(FACTORIES[name](params), train, args.fit_rows)
        seconds = perf_counter() - start
        row = dict(model=name, variant=variant, parameters=params,
                   cv_macro_f1=float(np.mean(folds_scores)), cv_fold_macro_f1=folds_scores,
                   train_seconds=seconds, **score(model, test))
        results.append(row)
        print(name, variant, row['cv_macro_f1'], row['top1_accuracy'], flush=True)
        return model

    for name, params in DEFAULTS.items():
        folds_scores = cross_validate(FACTORIES[name], params, train, folds, args.fit_rows)
        evaluate(name, params, folds_scores, 'baseline')
    chosen = max(results, key=lambda row: row['cv_macro_f1'])
    name = chosen['model']
    search_start = perf_counter()
    history = bayesian_search(name, train, folds, args.fit_rows, args.trials)
    search_seconds = perf_counter() - search_start
    best = max(history, key=lambda trial: trial['mean_cv_macro_f1'])
    tuned = evaluate(name, best['parameters'], best['fold_macro_f1'], 'tuned')
    tuned_is_final = best['mean_cv_macro_f1'] >= chosen['cv_macro_f1']
    final_model = tuned if tuned_is_final else fit(FACTORIES[name](chosen['parameters']), train, args.fit_rows)
    root = Path(__file__).resolve().parents[1]
    results_dir, artifacts_dir = root / 'results', root / 'artifacts'
    results_dir.mkdir(exist_ok=True)
    artifacts_dir.mkdir(exist_ok=True)
    metadata = dict(seed=SEED, k=K, features=FEATURES, fit_row_limit=args.fit_rows,
                    python=platform.python_version(), numpy=np.__version__,
                    scipy=scipy.__version__, scikit_learn=sklearn.__version__,
                    train_trips=len(train), student_test_trips=len(test),
                    validation_trip_ids=[[ids[i] for i in b] for _, b in folds],
                    optuna_version=optuna.__version__, tuning_sampler="TPE",
                    tuning_model=name, tuning_seconds=search_seconds,
                    search_space=SEARCH_SPACES[name],
                    final_variant='tuned' if tuned_is_final else 'baseline')
    (results_dir / 'experiments.json').write_text(json.dumps(dict(metadata=metadata, results=results, trials=history), indent=2))
    with (results_dir / 'comparison.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    joblib.dump(final_model, artifacts_dir / 'candidate_model.joblib', compress=3)
    (artifacts_dir / 'model_config.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
