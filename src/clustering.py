import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict


def gc_content(sequence):
    """Calculate GC content of a nucleotide sequence."""
    g = sequence.count('G')
    c = sequence.count('C')
    return (g + c) / len(sequence) * 100


def sequence_length(sequence):
    """Return the length of the sequence."""
    return len(sequence)


def sequence_complexity(sequence):
    """Calculate the complexity of a nucleotide sequence."""
    return len(set(sequence)) / len(sequence)


def kmer_signatures(sequence, k):
    """Generate k-mer signatures for a sequence."""
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]


def extract_features(sequences, k):
    """Extract features from a list of sequences."""
    features = []
    for seq in sequences:
        features.append([
            gc_content(seq),
            sequence_length(seq),
            sequence_complexity(seq)  
        ] + kmer_signatures(seq, k))
    return np.array(features)


def kmeans_clustering(features, n_clusters):
    """Perform K-means clustering on the features."""
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(features)
    return kmeans.labels_


def report_cluster_statistics(labels, sequences):
    """Report statistics of each cluster."""
    clusters = defaultdict(list)
    for label, seq in zip(labels, sequences):
        clusters[label].append(seq)
    
    for label, grouped_seqs in clusters.items():
        avg_gc = np.mean([gc_content(seq) for seq in grouped_seqs])
        print(f'Cluster {label}: {len(grouped_seqs)} sequences, Avg GC content: {avg_gc:.2f}%')


def group_within_cluster(labels, sequences):
    """Group sequences by their cluster labels."""
    clusters = defaultdict(list)
    for label, seq in zip(labels, sequences):
        clusters[label].append(seq)
    return dict(clusters)