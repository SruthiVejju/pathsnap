# pathsnap

## Data and feature pipeline

Part A reads one trip folder containing `.track`, `.nodes`, `.arcs`, and (for
labelled development data) `.route` files. It projects each trip using the UTM
zone selected from that trip's GPS coordinates, generates up to 25 road-arc
candidates within 60 m per GPS point, and aligns the ordered reference route
to create candidate labels. Heading and speed features do not bridge GPS gaps
longer than five seconds.

From the repository root in PowerShell, with the dataset in the sibling
`Student folder` directory:

```powershell
.\.venv\Scripts\python.exe scripts\eda.py '..\Student folder' results\eda
.\.venv\Scripts\python.exe scripts\build_dataset.py '..\Student folder\train' results\train.csv.gz
.\.venv\Scripts\python.exe scripts\build_dataset.py '..\Student folder\student_test' results\student_test.csv.gz
```

The candidate table includes `trip_id`, `point_idx`, `arc_id`, candidate
geometry/motion features, and `label`. `*_miss.csv` records the per-trip
candidate miss rate and aligned-route distance. Keep evaluator-testing data
and labels out of this pipeline during development.

To produce an additional training table for sensitivity checks, exclude
specific flagged trips explicitly; their audit metrics remain in the miss
report, and their IDs/reasons are written to `*_excluded.csv`:

```powershell
.\.venv\Scripts\python.exe scripts\build_dataset.py '..\Student folder\train' results\train_exclude_33_54.csv.gz --exclude-trip 00000033 --exclude-trip 00000054
```