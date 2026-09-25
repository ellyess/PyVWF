"""KMeans partitions do not depend on the machine's thread count.

scikit-learn's KMeans sums over OpenMP threads, and on the synthetic NL
sampling grid 1 and 8 threads gave one labelling while 2 and 4 gave two others:
a near-tie falls the other way when the reduction order changes. Until
2026-09-24 a side effect of importing the removed legacy module set every
process to one thread, which is what hid it; CI's four-core runners exposed it
once that module went. The KMeans calls now run on one thread, so the partition
is the same whatever the caller's thread settings. This file checks the one in
``pyvwf.sampling.cluster_with_geometries``, which builds the sampling grid; the
guard in ``pyvwf.clustering.cluster_turbines`` has no thread-count test.
"""

from __future__ import annotations

import pytest
from threadpoolctl import threadpool_limits

from pyvwf.sampling import cluster_with_geometries, create_sampling_points
from pyvwf.datasets.country_grid import COUNTRY_CONFIGS


def nl_labels():
    config = COUNTRY_CONFIGS["NL"]
    points = create_sampling_points(
        country_bounds=config["bounds"],
        method="grid",
        resolution=config["grid_resolution"],
        add_metadata=True,
        default_height=config["height"],
        default_model=config["model"],
        default_capacity=config["capacity"],
    )
    clustered, _ = cluster_with_geometries(
        sampling_points=points,
        num_clusters=config["num_clusters"],
        method="kmeans",
        country_code="NL",
        cluster_mode="onshore",
        geometry_type="voronoi",
    )
    return clustered["cluster"].to_numpy()


@pytest.mark.parametrize("threads", [2, 4])
def test_the_partition_is_the_same_on_any_thread_count(threads):
    with threadpool_limits(limits=1):
        reference = nl_labels()
    with threadpool_limits(limits=threads):
        assert (nl_labels() == reference).all()
