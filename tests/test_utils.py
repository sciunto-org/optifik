import unittest
from optifik.utils import round_to_uncertainty

class TestRoundToUncertainty(unittest.TestCase):

    def test_basic_functionality_1_digit(self):
        """Test basic functionality with 1 significant digit"""
        # Test case 1: 12.34567 ± 0.0234 -> 12.34 ± 0.02
        result_val, result_unc = round_to_uncertainty(12.34567, 0.0234, 1)
        self.assertEqual(result_val, "12.35")
        self.assertEqual(result_unc, "0.02")

        # Test case 2: 12.34567 ± 0.234 -> 12.3 ± 0.2
        result_val, result_unc = round_to_uncertainty(12.34567, 0.234, 1)
        self.assertEqual(result_val, "12.3")
        self.assertEqual(result_unc, "0.2")

        # Test case 3: 12.34567 ± 2.34 -> 12 ± 2
        result_val, result_unc = round_to_uncertainty(12.34567, 2.34, 1)
        self.assertEqual(result_val, "12")
        self.assertEqual(result_unc, "2")

        # Test case 4: 12.34567 ± 0.00234 -> 12.345 ± 0.002
        result_val, result_unc = round_to_uncertainty(12.34567, 0.00234, 1)
        self.assertEqual(result_val, "12.346")
        self.assertEqual(result_unc, "0.002")

    def test_basic_functionality_2_digits(self):
        """Test basic functionality with 2 significant digits"""
        # Test case 1: 12.34567 ± 0.0234 -> 12.345 ± 0.023
        result_val, result_unc = round_to_uncertainty(12.34567, 0.0234, 2)
        self.assertEqual(result_val, "12.346")
        self.assertEqual(result_unc, "0.023")

        # Test case 2: 12.34567 ± 0.234 -> 12.34 ± 0.23
        result_val, result_unc = round_to_uncertainty(12.34567, 0.234, 2)
        self.assertEqual(result_val, "12.35")
        self.assertEqual(result_unc, "0.23")

        # Test case 3: 12.34567 ± 2.34 -> 12.3 ± 2.3
        result_val, result_unc = round_to_uncertainty(12.34567, 2.34, 2)
        self.assertEqual(result_val, "12.3")
        self.assertEqual(result_unc, "2.3")

        # Test case 4: 12.34567 ± 0.00234 -> 12.3456 ± 0.0023
        result_val, result_unc = round_to_uncertainty(12.34567, 0.00234, 2)
        self.assertEqual(result_val, "12.3457")
        self.assertEqual(result_unc, "0.0023")

    def test_basic_functionality_3_digits(self):
        """Test basic functionality with 3 significant digits"""
        # Test case 1: 12.34567 ± 0.0234 -> 12.3457 ± 0.0234
        result_val, result_unc = round_to_uncertainty(12.34567, 0.0234, 3)
        self.assertEqual(result_val, "12.3457")
        self.assertEqual(result_unc, "0.0234")

        # Test case 2: 12.34567 ± 0.234 -> 12.346 ± 0.234
        result_val, result_unc = round_to_uncertainty(12.34567, 0.234, 3)
        self.assertEqual(result_val, "12.346")
        self.assertEqual(result_unc, "0.234")

        # Test case 3: 12.34567 ± 2.34 -> 12.35 ± 2.34
        result_val, result_unc = round_to_uncertainty(12.34567, 2.34, 3)
        self.assertEqual(result_val, "12.35")
        self.assertEqual(result_unc, "2.34")

        # Test case 4: 12.34567 ± 0.00234 -> 12.34567 ± 0.00234
        result_val, result_unc = round_to_uncertainty(12.34567, 0.00234, 3)
        self.assertEqual(result_val, "12.34567")
        self.assertEqual(result_unc, "0.00234")

    def test_edge_cases(self):
        """Test edge cases and special values"""
        # Zero uncertainty
        result_val, result_unc = round_to_uncertainty(12.3456, 0)
        self.assertEqual(result_val, 12.3456)
        self.assertEqual(result_unc, 0)

        # Integer values
        result_val, result_unc = round_to_uncertainty(100, 5.67, 1)
        self.assertEqual(result_val, "100")
        self.assertEqual(result_unc, "6")

        # Very small uncertainty
        result_val, result_unc = round_to_uncertainty(1.23456789, 0.000123, 2)
        self.assertEqual(result_val, "1.23457")
        self.assertEqual(result_unc, "0.00012")

    def test_rounding_behavior(self):
        """Test specific rounding behaviors"""
        # Test rounding up
        result_val, result_unc = round_to_uncertainty(12.345, 0.056, 1)
        self.assertEqual(result_val, "12.35")
        self.assertEqual(result_unc, "0.06")

        # Test rounding down
        result_val, result_unc = round_to_uncertainty(12.344, 0.054, 1)
        self.assertEqual(result_val, "12.34")
        self.assertEqual(result_unc, "0.05")

        # Test exact half (should round up)
        result_val, result_unc = round_to_uncertainty(12.345, 0.05, 1)
        self.assertEqual(result_val, "12.35")
        self.assertEqual(result_unc, "0.05")

    def test_different_input_types(self):
        """Test that function works with different input types"""
        # Float inputs
        result_val, result_unc = round_to_uncertainty('12.3456', '0.0234', 1)
        self.assertIsInstance(result_val, str)
        self.assertIsInstance(result_unc, str)

        # Float inputs
        result_val, result_unc = round_to_uncertainty(12.3456, 0.0234, 1)
        self.assertIsInstance(result_val, str)
        self.assertIsInstance(result_unc, str)

        # Integer inputs
        result_val, result_unc = round_to_uncertainty(100, 25, 1)
        self.assertEqual(result_val, "100")
        self.assertEqual(result_unc, "30")


    def test_very_large_uncertainty(self):
        """Test cases with uncertainty larger than value"""
        result_val, result_unc = round_to_uncertainty(12.3456, 25.67, 1)
        self.assertEqual(result_val, "12")
        self.assertEqual(result_unc, "30")

        result_val, result_unc = round_to_uncertainty(1.234, 10.5, 1)
        self.assertEqual(result_val, "1")
        self.assertEqual(result_unc, "10")

    def test_default_parameter(self):
        """Test that default parameter (1 digit) works correctly"""
        result_val, result_unc = round_to_uncertainty(12.34567, 0.0234)
        self.assertEqual(result_val, "12.35")
        self.assertEqual(result_unc, "0.02")

        result_val, result_unc = round_to_uncertainty(12.34567, 2.34)
        self.assertEqual(result_val, "12")
        self.assertEqual(result_unc, "2")

    def test_consistency_across_runs(self):
        """Test that the function produces consistent results"""
        # Multiple calls with same input should produce same output
        results = []
        for _ in range(5):
            result = round_to_uncertainty(12.34567, 0.0234, 2)
            results.append(result)

        # All results should be identical
        for i in range(1, len(results)):
            self.assertEqual(results[0], results[i])


# Additional test class for specific scenarios
#class TestRoundToUncertaintyAdvanced(unittest.TestCase):
#
#    def test_powers_of_ten(self):
#        """Test with values that are powers of ten"""
#        # Large numbers
#        result_val, result_unc = round_to_uncertainty(1234.56, 123.45, 1)
#        self.assertEqual(result_val, "1200")
#        self.assertEqual(result_unc, "100")
#
#        # Small numbers
#        result_val, result_unc = round_to_uncertainty(0.001234, 0.000123, 1)
#        self.assertEqual(result_val, "0.00123")
#        self.assertEqual(result_unc, "0.0001")
#
#    def test_boundary_conditions(self):
#        """Test boundary conditions for rounding"""
#        # Value near rounding boundary
#        result_val, result_unc = round_to_uncertainty(9.999, 0.0999, 1)
#        self.assertEqual(result_val, "10.00")
#        self.assertEqual(result_unc, "0.1")
#
#        # Uncertainty near rounding boundary
#        result_val, result_unc = round_to_uncertainty(12.345, 0.099, 1)
#        self.assertEqual(result_val, "12.35")
#        self.assertEqual(result_unc, "0.1")
#
