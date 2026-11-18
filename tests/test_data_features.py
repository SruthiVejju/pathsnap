import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pathsnap.candidates import point_heading
from pathsnap.data.geo import make_projector, project, utm_epsg


class GeoProjectionTests(unittest.TestCase):
    def test_selects_utm_zone_from_reference_coordinates(self):
        self.assertEqual(utm_epsg(8.6, 49.8), 32632)
        self.assertEqual(utm_epsg(-78.2, 0.3), 32617)
        self.assertEqual(utm_epsg(20.0, -10.0), 32734)

    def test_projects_coordinates_with_the_shared_transformer(self):
        lonlat = np.array([[8.6, 49.8], [8.61, 49.8]])
        transformer = make_projector(lonlat)
        xy = project(transformer, lonlat)
        self.assertEqual(xy.shape, (2, 2))
        self.assertTrue(np.isfinite(xy).all())
        self.assertGreater(np.linalg.norm(xy[1] - xy[0]), 0)


class HeadingGapTests(unittest.TestCase):
    def test_heading_and_speed_do_not_bridge_large_time_gaps(self):
        points = np.array([[0.0, 0.0], [2.0, 0.0], [100.0, 0.0]])
        timestamps = np.array([0.0, 1.0, 100.0])

        heading, step, speed = point_heading(points, timestamps, max_gap_s=5.0)

        np.testing.assert_allclose(heading[1], [1.0, 0.0])
        self.assertAlmostEqual(speed[1], 2.0)
        self.assertEqual(step[2], 0.0)
        self.assertEqual(speed[2], 0.0)


if __name__ == "__main__":
    unittest.main()
