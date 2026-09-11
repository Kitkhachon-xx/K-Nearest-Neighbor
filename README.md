# K-Nearest-Neighbor (KNN) — 2D Classification Example

A small, from-scratch implementation of the **K-Nearest Neighbors (KNN)** algorithm in Python + NumPy, built to teach the two core ideas behind KNN:

1. **Distance comparison** — measuring how "close" a new point is to every known point.
2. **Majority vote** — letting the closest points "vote" on what class the new point belongs to.

No `scikit-learn`, no black box — every step is plain NumPy so you can read and trace exactly what the algorithm does. The dataset, the test point, and `k` are all driven by a **YAML config file**, so you can experiment without touching a single line of Python.

---

## 1. What is KNN?

K-Nearest Neighbors is a **classification algorithm** based on a simple idea:

> "You are the average of the people (points) closest to you."

Given a new, unlabeled point, KNN:

1. Measures the distance from that point to **every** point in the training data.
2. Picks the **k** points with the smallest distance (the "nearest neighbors").
3. Looks at the **class label** of those k neighbors and predicts the label that appears **most often** (majority vote).

KNN doesn't "train" a model in the usual sense — there's no equation being fitted. It just remembers all the training data and does the distance + vote calculation at prediction time. This is called a **lazy learner** / **instance-based learner**.

---

## 2. The Dataset (2D Points)

Every data point in this project has:

| Field | Meaning |
| --- | --- |
| `x` | Position on the X axis (a feature, e.g. "hours streamed per week") |
| `y` | Position on the Y axis (a feature, e.g. "hours competing per week") |
| `label` | The class the point belongs to (`0`, `1`, `2`, ...) |

Because each point only has two features (`x`, `y`), we can plot every point on a normal 2D scatter plot — that's what makes this project easy to *see and understand* visually, not just read as numbers.

The example dataset represents three types of gamers, and now lives in **`config_dev.yaml`** instead of being hardcoded in `main.py`:

```yaml
collect_data:
  x: [1, 2, 3, 4]
  y: [0.5, 1, 0, 2.5]
  data_labels: [0, 1, 0, 2]
  labels_name: {0: "Streamer", 1: "Gamer", 2: "E-Sports"}
```

| Point | x | y | label | class name |
| --- | --- | --- | --- | --- |
| 0 | 1 | 0.5 | 0 | Streamer |
| 1 | 2 | 1.0 | 1 | Gamer |
| 2 | 3 | 0.0 | 0 | Streamer |
| 3 | 4 | 2.5 | 2 | E-Sports |

---

## 3. Step-by-Step: How the Algorithm Works Here

### Step 1 — Calculate distance (`calculate_distance`)

For a new point `(point_x, point_y)`, we compute the **Euclidean distance** to every training point:

```text
distance = √( (x_i - point_x)² + (y_i - point_y)² )
```

```python
# Process/Calculate.py
def calculate_distance(self, point_x, point_y):
    distances = np.sqrt((self.data_x - point_x) ** 2 + (self.data_y - point_y) ** 2)
    return distances
```

Thanks to NumPy vectorization, this line calculates the distance to **all** points at once (no `for` loop needed).

### Step 2 — Find the k nearest neighbors (`get_k_nearest_neighbors`)

```python
def get_k_nearest_neighbors(self, point_x, point_y, k):
    distances = self.calculate_distance(point_x, point_y)
    nearest_indices = np.argsort(distances)[:k]
    return self.data_label[nearest_indices]
```

- `np.argsort(distances)` sorts the **indices** of the points by distance, smallest first.
- `[:k]` keeps only the first `k` — i.e. the indices of the `k` closest points.
- `self.data_label[nearest_indices]` looks up the class label of each of those neighbors.

### Step 3 — Majority vote (`predict_label`)

```python
def _predict_numeric_label(self, point_x, point_y, k):
    neighbors_labels = self.get_k_nearest_neighbors(point_x, point_y, k)
    unique_labels, counts = np.unique(neighbors_labels, return_counts=True)
    majority_label = unique_labels[np.argmax(counts)]
    return majority_label
```

- `np.unique(..., return_counts=True)` counts how many times each label shows up among the k neighbors.
- `np.argmax(counts)` finds which label has the **highest count** — the "winner" of the vote.
- That winning label is the prediction.

`predict_label()` then wraps this numeric result and converts it to a human-readable class name using `label_names` (e.g. `0 → "Streamer"`), so predictions read as `"Gamer"` instead of `1`.

> ℹ️ Tie-breaking: if two labels get the same number of votes, `np.unique` returns labels sorted from smallest to largest, so `np.argmax` picks the **first** (numerically smallest) label among the tied ones.

---

## 4. Worked Example (the numbers behind `config_dev.yaml`)

Predicting the class of test point **(2.5, 2)** with **k = 3** (both come from `input_data` in the config), using the dataset above:

| Point | (x, y) | label | distance to (2.5, 2) |
| --- | --- | --- | --- |
| 0 | (1, 0.5) | 0 (Streamer) | √4.50 ≈ **2.121** |
| 1 | (2, 1.0) | 1 (Gamer) | √1.25 ≈ **1.118** ← closest |
| 2 | (3, 0.0) | 0 (Streamer) | √4.25 ≈ **2.062** |
| 3 | (4, 2.5) | 2 (E-Sports) | √2.50 ≈ **1.581** |

Sorted by distance (nearest → farthest): **Point 1 (1.118) → Point 3 (1.581) → Point 2 (2.062) → Point 0 (2.121)**

With `k = 3`, the 3 nearest neighbors are **Point 1, Point 3, Point 2** → labels `[1, 2, 0]`.

Vote count: label `0` → 1 vote, label `1` → 1 vote, label `2` → 1 vote — a **3-way tie**.
Since ties resolve to the smallest label value, the result is **label `0` → "Streamer"**.

Running `main.py` prints exactly this:

```text
Predicted label for point (2.5, 2) with k=3: Streamer
```

Try changing `k` in `config_dev.yaml` (e.g. `k: 1`) to see the vote — and the prediction — change.

---

## 5. Project Structure

```text
K-Nearest-Neighbor/
├── main.py                  # Entry point: loads config, builds the dataset, runs prediction, shows plots
├── config_dev.yaml          # Dataset + test point + k — edit this to experiment, no code changes needed
├── requirements.txt         # Python dependencies
├── Load_utils/
│   └── load_config.py       # load_config(): reads a YAML file into a dict
└── Process/
    ├── Data.py               # Data: holds the dataset (x, y, label) as a simple container
    ├── Calculate.py           # KNNCalculate: the KNN algorithm itself (distance + vote)
    └── Plot.py                # Plot: all matplotlib visualization code
```

Each piece has one job (separation of concerns):

| Module | File | Responsibility |
| --- | --- | --- |
| `load_config` | `Load_utils/load_config.py` | Load `config_dev.yaml` into a plain `dict` |
| `Data` | `Process/Data.py` | Store the dataset points and their colors/labels |
| `KNNCalculate` | `Process/Calculate.py` | Distance calculation, nearest-neighbor lookup, majority-vote prediction |
| `Plot` | `Process/Plot.py` | Draw the raw data scatter plot and the KNN decision boundary |

---

## 6. Configuration (`config_dev.yaml`)

The whole point of the config file is: **change the data, the test point, or `k` without touching `main.py`.**

```yaml
collect_data:
  x: [1, 2, 3, 4]
  y: [0.5, 1, 0, 2.5]
  data_labels: [0, 1, 0, 2]
  labels_name: {0: "Streamer", 1: "Gamer", 2: "E-Sports"}

input_data:
  x: 2.5
  y: 2
  k: 3
```

| Section | Key | Meaning |
| --- | --- | --- |
| `collect_data` | `x`, `y` | The training points' coordinates (must be the same length) |
| `collect_data` | `data_labels` | The class label (integer) for each training point, in the same order as `x`/`y` |
| `collect_data` | `labels_name` | Maps each numeric label to a human-readable class name, used by `predict_label()` |
| `input_data` | `x`, `y` | The test point to classify |
| `input_data` | `k` | How many nearest neighbors to vote with |

`main.py` reads these with a safe, nested `.get(...)` pattern:

```python
config = load_config("config_dev.yaml")
data_x = np.array(config.get("collect_data", {}).get("x", [1, 2, 3, 4]))
test_point_x = config.get("input_data", {}).get("x", 2.5)
k = config.get("input_data", {}).get("k", 3)
```

This means every value has a hardcoded fallback baked into `main.py` — if a key (or the whole `collect_data`/`input_data` section) is missing from the YAML, that default is used instead of crashing.

### How `load_config()` works

```python
# Load_utils/load_config.py
def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8-sig") as config_file:
        config = yaml.safe_load(config_file)
    return config or {}
```

- Opens the YAML file with `utf-8-sig` encoding (so it still works even if the file was saved with a BOM, e.g. from some Windows editors/Excel).
- `yaml.safe_load()` parses the YAML text into a Python `dict`.
- **`return config or {}`** — important safety net: if the YAML file is empty (no content at all), `yaml.safe_load()` returns `None`, not `{}`. Without this fallback, the very next line in `main.py` (`config.get("collect_data", ...)`) would crash with `AttributeError: 'NoneType' object has no attribute 'get'`. Returning `{}` instead means every `.get(key, default)` call in `main.py` just falls through to its hardcoded default — no crash, no config needed to run the demo.

> ⚠️ Common mistake: if you accidentally clear out `config_dev.yaml`, the project still runs (thanks to the fallback above) — it will just quietly use the built-in defaults instead of your edited values. If your changes don't seem to take effect, double check `config_dev.yaml` still has your content and correct YAML indentation.

### Changing the experiment — no Python required

- **Change `k`:** edit `input_data.k` in `config_dev.yaml`.
- **Move the test point:** edit `input_data.x` / `input_data.y`.
- **Add or change training points:** edit `collect_data.x`, `collect_data.y`, `collect_data.data_labels` (keep all three the same length).
- **Rename classes:** edit `collect_data.labels_name`.
- **Run a different dataset entirely:** copy `config_dev.yaml` to e.g. `config_prod.yaml`, then change the filename in `main.py`'s `load_config("config_dev.yaml")` call.

---

## 7. Setup & Running

### Install dependencies

```bash
pip install -r requirements.txt
```

This installs `numpy`, `matplotlib`, and `PyYAML` (used by `load_config()` to parse `config_dev.yaml`).

### Run the example

```bash
python main.py
```

> Run this from inside the `K-Nearest-Neighbor/` folder — `load_config("config_dev.yaml")` uses a relative path, so it looks for the file in the current working directory.

This will:

1. Load `config_dev.yaml` (or fall back to built-in defaults if a value is missing).
2. Print each training point and its class name.
3. Open a scatter plot of the raw data (`Plot.plot_data`).
4. Open a KNN **decision boundary** plot (`Plot.plot_decision_boundary`) — background color shows what class KNN would predict for *any* point in that region, given the current `k`.
5. Print the predicted class name for the test point.

---

## 8. Understanding the Plots

### `Plot.plot_data(data)`

A simple scatter plot of the training points, colored by class. This is just "what the data looks like" before any prediction happens.

### `Plot.plot_decision_boundary(knn, k, test_point=...)`

This is the most useful plot for *understanding* KNN:

- It builds a grid covering the whole 2D space and asks the KNN model to predict a label for **every** grid cell → this produces the colored background regions ("decision boundary"). Wherever the background color changes, that's where KNN's prediction flips from one class to another.
- The **training points** are drawn as a scatter on top, so you can see how the boundary follows the data.
- If you pass `test_point=(x, y)`, it also draws:
  - The test point as a **yellow star** ⭐, labeled with its predicted class name.
  - **Dashed lines** connecting the test point to its `k` nearest neighbors — literally showing you which points "voted".
  - **Black circles** highlighting those k nearest neighbor points.

This turns the abstract "distance + vote" math into something you can see directly: the star, the lines, and which neighbors get circled *are* the algorithm.

---

## 9. Quick API Reference

**`load_config(config_path: str) -> dict`** (`Load_utils/load_config.py`)
Reads a YAML file and returns it as a `dict` (empty `dict` if the file has no content).

**`Data(data_x, data_y, data_label, colors=['red','blue','green'])`**
Simple container for the dataset. `len(data)` returns the number of points.

**`KNNCalculate(data_x, data_y, data_label, label_names=None)`**

- `calculate_distance(x, y)` → distance from `(x, y)` to every training point.
- `get_k_nearest_neighbors(x, y, k)` → labels of the k closest training points.
- `predict_label(x, y, k)` → predicted class (name if `label_names` was provided, otherwise the numeric label).

**`Plot`** (static methods, take a `Data` or `KNNCalculate` instance)

- `Plot.plot_data(data)`
- `Plot.plot_decision_boundary(knn, k, x_range=None, y_range=None, resolution=100, test_point=None)`

---

## 10. Things to Try (Exercises)

These are good ways to build intuition about KNN using this exact codebase — all doable by editing **`config_dev.yaml` only**, no Python required:

1. **Change `input_data.k`** (try `1`, `2`, `4`) and see how the prediction and the decision boundary shape change. Small `k` → boundary follows individual points closely (can overfit / be noisy). Large `k` → boundary gets smoother but may ignore local structure.
2. **Add more points** to `collect_data.x` / `collect_data.y` / `collect_data.data_labels` and re-run — watch the decision boundary regions reshape.
3. **Move the test point** (`input_data.x`, `input_data.y`) around and watch which neighbors light up and how the predicted class changes.
4. **Force a tie** on purpose (like the worked example above) and confirm which label wins, to understand the tie-breaking rule.
5. **Rename `collect_data.labels_name`** to match a real classification problem you care about (e.g. billing tiers, customer segments) instead of Streamer/Gamer/E-Sports.
6. **Create a second config file** (e.g. `config_prod.yaml`) with a different dataset, then point `main.py`'s `load_config("config_dev.yaml")` call at it to switch experiments without editing any other code.

---

## 11. Limitations (by design, for learning)

This implementation is intentionally minimal so the algorithm stays readable — it does **not** include things a production KNN would need:

- No feature scaling/normalization — since `x` and `y` are on similar scales here, this doesn't matter for the demo, but with real, differently-scaled features you'd normally standardize features first (distance would otherwise be dominated by whichever feature has the larger numeric range).
- Only works with 2 features (`x`, `y`) — real KNN generalizes to any number of features/dimensions.
- Only Euclidean distance is implemented (other options include Manhattan or cosine distance).
- `k` is set manually in the config rather than tuned automatically (e.g. via cross-validation).

These are great follow-up topics once you're comfortable with the basic distance + majority-vote mechanics shown here.
