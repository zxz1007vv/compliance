import unittest

from wbc_compliance_gym.utils.terrain_proportions import (
    BOX_TERRAIN_TYPES,
    HEIGHTFIELD_TERRAIN_TYPES,
    cumulative_terrain_proportions,
)


class TerrainProportionTests(unittest.TestCase):
    def test_heightfield_and_box_type_schemas_are_independent(self):
        self.assertEqual(10, len(HEIGHTFIELD_TERRAIN_TYPES))
        self.assertEqual(5, len(BOX_TERRAIN_TYPES))
        self.assertEqual("uniform_roughness", HEIGHTFIELD_TERRAIN_TYPES[8])
        self.assertEqual("pit", BOX_TERRAIN_TYPES[4])

    def test_valid_probabilities_are_converted_to_cumulative_thresholds(self):
        cumulative = cumulative_terrain_proportions(
            [0.2, 0.3, 0.5],
            ("first", "second", "third"),
            "terrain.test_proportions",
        )
        self.assertEqual([0.2, 0.5, 1.0], cumulative)

    def test_probability_count_is_validated(self):
        with self.assertRaisesRegex(ValueError, "must contain 5 probabilities"):
            cumulative_terrain_proportions(
                [1.0], BOX_TERRAIN_TYPES, "terrain.box_terrain_proportions"
            )

    def test_negative_probability_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            cumulative_terrain_proportions(
                [1.1, -0.1, 0.0, 0.0, 0.0],
                BOX_TERRAIN_TYPES,
                "terrain.box_terrain_proportions",
            )

    def test_probabilities_must_sum_to_one(self):
        with self.assertRaisesRegex(ValueError, "must sum to 1.0"):
            cumulative_terrain_proportions(
                [0.1, 0.1, 0.1, 0.1, 0.1],
                BOX_TERRAIN_TYPES,
                "terrain.box_terrain_proportions",
            )


if __name__ == "__main__":
    unittest.main()
