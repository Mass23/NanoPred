def test_clustering_effectiveness(clusters, features):
    """Evaluate clustering effectiveness using metrics like Silhouette Score."""
    from sklearn.metrics import silhouette_score
    import numpy as np
    
    if len(set(clusters)) <= 1:
        return "Not enough clusters to evaluate."
    
    score = silhouette_score(features, clusters)
    return score


def heuristic_filtering(data, heuristic):
    """Apply heuristic filtering on the given data."""
    filtered_data = []
    for item in data:
        if heuristic(item):
            filtered_data.append(item)
    return filtered_data
