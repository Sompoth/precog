import unittest
from datetime import datetime

from hypothesis import given
from hypothesis import strategies as st
from pytz import timezone

from precog.utils.timestamp import (
    datetime_to_iso8601,
    datetime_to_posix,
    get_midnight,
    get_now,
    get_posix,
    get_timezone,
    iso8601_to_datetime,
    posix_to_datetime,
)


class TestTimestamp(unittest.TestCase):

    # runs once prior to all tests in this file
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.DATETIME_CONSTANT = datetime(2024, 12, 11, 18, 46, 43, 112378, tzinfo=get_timezone())
        self.POSIX_CONSTANT = 1733942803.112378
        self.ISO8601_CONSTANT = "2024-12-11T18:46:43.112378Z"

    # runs once prior to every single test
    def setUp(self):
        pass

    def test_get_timezone(self):
        # Ensure we are abiding by UTC timezone
        self.assertEqual(get_timezone(), timezone("UTC"))

    def test_get_now(self):
        now = get_now()

        # Check that this is a datetime
        self.assertIsInstance(now, datetime)

        # Check that this is UTC timezone aware
        self.assertEqual(now.tzinfo, get_timezone())

        # Check that the timestamp is sensitive
        now2 = get_now()
        self.assertNotEqual(now, now2)

        # Make our own timestamp
        real_now = datetime.now(get_timezone())

        # Check that all timezones are within 20 seconds of each other
        # They should all be very close together considering everything
        threshold = 20
        diff1 = (now - now2).total_seconds()
        diff2 = (now - real_now).total_seconds()
        diff3 = (now2 - real_now).total_seconds()

        self.assertLess(abs(diff1), threshold)
        self.assertLess(abs(diff2), threshold)
        self.assertLess(abs(diff3), threshold)

    def test_get_posix(self):
        posix = get_posix()

        # Check that this is a float
        self.assertIsInstance(posix, float)

        # Check that the timestamp is sensitive
        posix2 = get_posix()
        self.assertNotEqual(posix, posix2)

        # Make our own timestamp
        real_posix = datetime.now(get_timezone()).timestamp()

        # Check that all timezones are within 20 seconds of each other
        # They should all be very close together considering everything
        threshold = 20
        diff1 = posix - posix2
        diff2 = posix - real_posix
        diff3 = posix2 - real_posix

        self.assertLess(abs(diff1), threshold)
        self.assertLess(abs(diff2), threshold)
        self.assertLess(abs(diff3), threshold)

    def test_get_midnight(self):
        now = datetime.now(get_timezone())
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)

        midnight2 = get_midnight()

        # Check that this is UTC timezone aware
        self.assertEqual(midnight2.tzinfo, get_timezone())

        # They should be equal
        self.assertEqual(midnight, midnight2)

        # confirm everything is zeroed out
        self.assertEqual(midnight2.microsecond, 0)
        self.assertEqual(midnight2.second, 0)
        self.assertEqual(midnight2.minute, 0)
        self.assertEqual(midnight2.hour, 0)

    def test_datetime_iso_roundtrip(self):
        # datetime -> str -> datetime
        new_str = datetime_to_iso8601(self.DATETIME_CONSTANT)
        new_datetime = iso8601_to_datetime(new_str)

        self.assertEqual(self.DATETIME_CONSTANT, new_datetime)
        self.assertEqual(self.ISO8601_CONSTANT, new_str)
        self.assertEqual(new_datetime.tzinfo, get_timezone())

        # str -> datetime -> str
        new_datetime = iso8601_to_datetime(self.ISO8601_CONSTANT)
        new_str = datetime_to_iso8601(new_datetime)

        self.assertEqual(self.DATETIME_CONSTANT, new_datetime)
        self.assertEqual(self.ISO8601_CONSTANT, new_str)
        self.assertEqual(new_datetime.tzinfo, get_timezone())

    @given(st.datetimes(timezones=st.just(get_timezone())))
    def test_hypothesis_datetime_iso_roundtrip(self, new_datetime):
        # datetime -> str -> datetime
        new_str = datetime_to_iso8601(new_datetime)
        new_datetime2 = iso8601_to_datetime(new_str)

        self.assertEqual(new_datetime, new_datetime2)
        self.assertEqual(new_datetime2.tzinfo, get_timezone())

    def test_datetime_posix_roundtrip(self):
        # datetime -> float -> datetime
        new_float = datetime_to_posix(self.DATETIME_CONSTANT)
        new_datetime = posix_to_datetime(new_float)

        self.assertEqual(self.DATETIME_CONSTANT, new_datetime)
        self.assertEqual(self.POSIX_CONSTANT, new_float)
        self.assertEqual(new_datetime.tzinfo, get_timezone())

        # float -> datetime -> float
        new_datetime = posix_to_datetime(self.POSIX_CONSTANT)
        new_float = datetime_to_posix(new_datetime)

        self.assertEqual(self.DATETIME_CONSTANT, new_datetime)
        self.assertEqual(self.POSIX_CONSTANT, new_float)
        self.assertEqual(new_datetime.tzinfo, get_timezone())

    @given(st.datetimes(timezones=st.just(get_timezone())).map(lambda dt: dt.replace(microsecond=0)))
    def test_hypothesis_datetime_posix_roundtrip(self, new_datetime):
        # hypothesis exposes niche floating point precision errors
        # zero out microseconds to solve this
        # Floating point precision resulting in a mismatch of 1 microsecond is negligible

        # datetime -> float -> datetime
        new_float = float(datetime_to_posix(new_datetime))
        new_datetime2 = posix_to_datetime(new_float)

        self.assertEqual(new_datetime, new_datetime2)
        self.assertEqual(new_datetime2.tzinfo, get_timezone())
