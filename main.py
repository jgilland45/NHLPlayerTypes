import fetching
import kmeans
import csv

def main():
    data = fetching.get_regular_stats()
    print(len(data))
    enhanced_data = fetching.get_edge_data(data)
    print(len(enhanced_data))

    result = kmeans.cluster(
        enhanced_data,
        n_clusters=None,
        output_path="cluster_plot.png",
        interactive_output_path="cluster_plot.html",
        show_plot=False,
    )
    print(f"Cluster count used: {result['n_clusters']}")
    print(f"Feature matrix shape: {result['X'].shape}")
    print(f"Cluster sizes: {result['cluster_counts']}")
    if result["plot_path"]:
        print(f"Cluster plot saved to: {result['plot_path']}")
    else:
        print("matplotlib not installed, so no plot was generated.")

    if result["interactive_plot_path"]:
        print(f"Interactive cluster plot (with player hover) saved to: {result['interactive_plot_path']}")
    else:
        print("plotly not installed, so interactive hover plot was not generated.")

    export_path = "clustering_input.csv"
    with open(export_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["player", "cluster", *result["feature_names"]]
        writer.writerow(header)

        for i, row in enumerate(result["X"]):
            player = result["player_labels"][i] if i < len(result["player_labels"]) else f"Player {i + 1}"
            cluster = int(result["labels"][i]) if i < len(result["labels"]) else ""
            writer.writerow([player, cluster, *row.tolist()])

    print(f"Clustering input data saved to: {export_path}")

if __name__ == "__main__":
    main()