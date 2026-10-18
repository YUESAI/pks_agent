import unittest
from scripts.probe_dates_motor import parse_date, shape
class DateMotorProbeTests(unittest.TestCase):
    def test_iso_timestamp_and_mask(self):
        self.assertEqual(str(parse_date("2025-01-01T12:23:45.012Z")), "2025-01-01")
        self.assertEqual(shape("2025-01-01"), "9999-99-99")
        self.assertIsNone(parse_date("unreadable"))
if __name__ == "__main__":
    unittest.main()
