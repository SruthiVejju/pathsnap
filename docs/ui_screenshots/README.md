# UI screenshots

Evaluator dashboard (Laptop 2) talking to the API (Laptop 1), showing the server-calculated metrics.

| File | What it shows |
|---|---|
| `ui_student_dropdown_trip_00000007.jpg` | a student-test trip chosen in the "Student test trip" dropdown |
| `ui_evaluator_dropdown_trip_00000059.jpg` | an evaluator trip chosen in the "Evaluator test trip" dropdown |

Each dropdown shows its own label as a placeholder while the other one is active. Accuracy, route continuity and route F1 are computed on the server against its private reference route; the UI only displays the numbers.
