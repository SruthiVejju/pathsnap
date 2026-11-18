"""Build candidate features and derive approximate development labels."""
import numpy as np
from scipy.spatial import cKDTree

FEATURES = ('distance_m', 'heading_cosine', 'segment_length_m', 'speed_mps', 'rank')


def project(points, origin):
    scale = np.array([np.cos(np.deg2rad(origin[1])), 1.0]) * 111195
    return (points - origin) * scale


def segment_distances(points, starts, vectors):
    delta = points[:, None, :] - starts
    length_sq = (vectors ** 2).sum(axis=-1)
    fraction = (delta * vectors).sum(axis=-1) / np.maximum(length_sq, 1e-12)
    return np.linalg.norm(delta - np.clip(fraction, 0, 1)[..., None] * vectors, axis=-1)


def candidates(track, nodes, arcs, k=5):
    """Return candidate arc IDs and feature rows, with no label input."""
    origin = track[0, :2]
    gps = project(track[:, :2], origin)
    ends = project(nodes, origin)[arcs]
    starts, vectors = ends[:, 0], ends[:, 1] - ends[:, 0]
    lengths = np.linalg.norm(vectors, axis=1)
    pieces = np.maximum(1, np.ceil(lengths / 20).astype(int))
    owners = np.repeat(np.arange(len(arcs)), pieces + 1)
    offsets = np.cumsum(pieces + 1) - pieces - 1
    fractions = (np.arange(len(owners)) - offsets[owners]) / pieces[owners]
    samples = starts[owners] + fractions[:, None] * vectors[owners]
    tree = cKDTree(samples)
    k = min(k, len(arcs))
    _, sample_ids = tree.query(gps, k=min(64, len(samples)))
    ids = np.empty((len(gps), k), dtype=int)
    for i, matches in enumerate(np.atleast_2d(sample_ids).reshape(len(gps), -1)):
        shortlist = np.unique(owners[matches])
        if len(shortlist) < k:
            shortlist = np.arange(len(arcs))
        distances = segment_distances(gps[i:i+1], starts[shortlist], vectors[shortlist])[0]
        ids[i] = shortlist[np.argsort(distances, kind='stable')[:k]]
    distance = segment_distances(gps, starts[ids], vectors[ids])
    displacement = np.diff(gps, axis=0, prepend=gps[:1])
    time = np.diff(track[:, 2], prepend=track[0, 2])
    movement = np.linalg.norm(displacement, axis=1)
    heading = (displacement[:, None, :] * vectors[ids]).sum(axis=-1)
    heading /= np.maximum(movement[:, None] * lengths[ids], 1e-12)
    speed = np.divide(movement, time, out=np.zeros_like(movement), where=time > 0)
    features = np.stack((distance, heading, lengths[ids],
                         np.broadcast_to(speed[:, None], ids.shape),
                         np.broadcast_to(np.arange(1, k + 1), ids.shape)), axis=-1)
    return ids, features.astype(np.float32)


def reference_labels(track, nodes, arcs, route):
    """Use the closest segment on the reference route as a pointwise proxy."""
    origin = track[0, :2]
    gps = project(track[:, :2], origin)
    ends = project(nodes, origin)[arcs[route]]
    starts, vectors = ends[:, 0], ends[:, 1] - ends[:, 0]
    labels = [route[segment_distances(batch, starts, vectors).argmin(axis=1)]
              for batch in np.array_split(gps, max(1, (len(gps) + 127) // 128))]
    return np.concatenate(labels)
