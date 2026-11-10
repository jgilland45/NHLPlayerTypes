import numpy as np
import importlib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PLAYER_TYPE_FEATURE_PRIORITY = [
    "goals",
    "assists",
    "points",
    "shots",
    "shootingPct",
    "faceoffWinPct",
    "timeOnIcePerGame",
    "plusMinus",
    "pointsPerGame",
    "shotsPerGame",
    "topShotSpeed",
    "skatingSpeed",
    "totalDistanceSkated",
    "distanceMaxGame",
    "offensiveZone",
    "defensiveZone",
    "neutralZone",
    "sogSummary",
    "sogDetails",
    "zoneTimeDetails",
    "speedMax",
    "burstsOver20",
    "locationCode",
    "shot",
    "shooting",
    "goalsPercentile",
    "shotsPercentile",
    "percentile",
    "leagueAvg",
]


def _is_number(value):
    return isinstance(value, (int, float, np.integer, np.floating, bool)) and not isinstance(value, str)


def _to_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("default", "en", "fr"):
            v = value.get(key)
            if isinstance(v, str) and v.strip():
                return v
        for v in value.values():
            if isinstance(v, str) and v.strip():
                return v
    return ""


def _player_label(row, idx):
    if not isinstance(row, dict):
        return f"Player {idx + 1}"

    for key in ("fullName", "name"):
        text = _to_text(row.get(key))
        if text:
            return text

    first = _to_text(row.get("firstName"))
    last = _to_text(row.get("lastName"))
    full = f"{first} {last}".strip()
    if full:
        return full

    player_id = row.get("playerId")
    if player_id is not None:
        return f"Player {player_id}"

    return f"Player {idx + 1}"


def _flatten_row(value, parent_key="", sep="."):
    flat = {}

    if isinstance(value, dict):
        for key, child in value.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
            flat.update(_flatten_row(child, new_key, sep=sep))
        return flat

    if isinstance(value, list):
        for i, child in enumerate(value):
            new_key = f"{parent_key}[{i}]" if parent_key else f"[{i}]"
            flat.update(_flatten_row(child, new_key, sep=sep))
        return flat

    flat[parent_key] = value
    return flat


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _convex_hull(points):
    if len(points) <= 1:
        return points

    pts = sorted(map(tuple, points.tolist()))

    lower = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    return np.array(hull, dtype=float)


def _to_matrix(input_data):
    if not input_data:
        return np.empty((0, 0), dtype=float), []

    # If it's already array-like numeric data, use it directly.
    if not isinstance(input_data[0], dict):
        X = np.asarray(input_data, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X, [f"feature_{i}" for i in range(X.shape[1])]

    # enhanced_data is typically nested and mixed type.
    # Flatten to one-level keys, then keep numeric-only features.
    flattened_rows = [_flatten_row(row) for row in input_data]
    numeric_rows = [
        {
            key: float(value)
            for key, value in row.items()
            if _is_number(value) and np.isfinite(value)
        }
        for row in flattened_rows
    ]

    all_numeric_keys = sorted({key for row in numeric_rows for key in row.keys()})
    feature_keys = _preferred_feature_keys(all_numeric_keys)

    X = np.full((len(input_data), len(feature_keys)), np.nan, dtype=float)

    for i, row in enumerate(numeric_rows):
        for j, key in enumerate(feature_keys):
            value = row.get(key)
            if value is not None:
                X[i, j] = value

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, feature_keys


def _preferred_feature_keys(all_numeric_keys):
    normalized = {key.lower(): key for key in all_numeric_keys}
    ordered = []
    seen = set()

    for token in PLAYER_TYPE_FEATURE_PRIORITY:
        token_lower = token.lower()
        for key in sorted(normalized):
            if key not in seen and token_lower in key:
                ordered.append(normalized[key])
                seen.add(key)

    for key in sorted(normalized):
        if key not in seen:
            ordered.append(normalized[key])
            seen.add(key)

    return ordered


def _choose_cluster_count(n_samples):
    if n_samples <= 0:
        return 1

    # Data-sized heuristic: k ≈ sqrt(n), with only a lower bound so the
    # clustering remains meaningful for very large inputs.
    k = int(round(np.sqrt(n_samples)))
    return max(1, k)


def cluster(
    input_data,
    n_clusters=None,
    random_state=0,
    output_path="cluster_plot.png",
    interactive_output_path="cluster_plot.html",
    show_plot=False,
):
    if n_clusters is None:
        n_clusters = _choose_cluster_count(len(input_data))

    n_clusters = max(2, min(int(n_clusters), max(2, len(input_data))))

    plt = None
    px = None
    go = None
    try:
        plt = importlib.import_module("matplotlib.pyplot")
    except ImportError:
        plt = None
    try:
        px = importlib.import_module("plotly.express")
    except ImportError:
        px = None
    try:
        go = importlib.import_module("plotly.graph_objects")
    except ImportError:
        go = None

    X, feature_names = _to_matrix(input_data)

    if X.size == 0:
        return {
            "X": X,
            "labels": np.array([], dtype=int),
            "feature_names": feature_names,
            "plot_path": None,
            "cluster_counts": {},
        }

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=random_state)
    X_2d = pca.fit_transform(X_scaled)
    centers_2d = pca.transform(kmeans.cluster_centers_)
    player_labels = [_player_label(row, i) for i, row in enumerate(input_data)]

    interactive_plot_path = None

    if px is not None:
        fig_data = {
            "x": X_2d[:, 0],
            "y": X_2d[:, 1],
            "cluster": labels.astype(str),
            "player": player_labels,
        }
        fig = px.scatter(
            fig_data,
            x="x",
            y="y",
            color="cluster",
            hover_name="player",
            hover_data={"cluster": True, "x": ":.3f", "y": ":.3f"},
            title=f"NHL Player Clusters (k={n_clusters})",
        )

        if go is not None:
            for cluster_id in np.unique(labels):
                cluster_points = X_2d[labels == cluster_id]
                if cluster_points.shape[0] < 3:
                    continue

                hull = _convex_hull(cluster_points)
                if hull.shape[0] < 3:
                    continue

                hull_closed = np.vstack([hull, hull[0]])
                fig.add_trace(
                    go.Scatter(
                        x=hull_closed[:, 0],
                        y=hull_closed[:, 1],
                        mode="lines",
                        fill="toself",
                        opacity=0.12,
                        line={"width": 1.5},
                        name=f"Cluster {int(cluster_id)} boundary",
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

            fig.add_trace(
                go.Scatter(
                    x=centers_2d[:, 0],
                    y=centers_2d[:, 1],
                    mode="markers",
                    marker={"symbol": "x", "size": 12, "color": "black"},
                    name="Centroids",
                    hovertemplate="Centroid<extra></extra>",
                )
            )

        fig.write_html(interactive_output_path, include_plotlyjs="cdn")
        interactive_plot_path = interactive_output_path

    if plt is not None:
        plt.figure(figsize=(11, 7))
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab20", s=26, alpha=0.8)
        plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c="black", marker="X", s=180, label="Centroids")

        for cluster_id in np.unique(labels):
            cluster_points = X_2d[labels == cluster_id]
            if cluster_points.shape[0] < 3:
                continue

            hull = _convex_hull(cluster_points)
            if hull.shape[0] < 3:
                continue

            hull_closed = np.vstack([hull, hull[0]])
            plt.plot(hull_closed[:, 0], hull_closed[:, 1], linewidth=1.2, alpha=0.7)
            plt.fill(hull_closed[:, 0], hull_closed[:, 1], alpha=0.08)

        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.title(f"NHL Player Clusters (k={n_clusters})")
        plt.legend()
        plt.colorbar(scatter, label="Cluster")
        plt.tight_layout()
        plt.savefig(output_path, dpi=140)

        if show_plot:
            plt.show()
        else:
            plt.close()

    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_counts = {int(label): int(count) for label, count in zip(unique_labels, counts)}

    return {
        "X": X,
        "labels": labels,
        "feature_names": feature_names,
        "player_labels": player_labels,
        "plot_path": output_path if plt is not None else None,
        "interactive_plot_path": interactive_plot_path,
        "cluster_counts": cluster_counts,
        "n_clusters": int(n_clusters),
    }