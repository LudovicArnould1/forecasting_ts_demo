"""Tests for utility functions."""

import pytest

from data_processing.utils import get_patch_size_for_freq, parse_frequency


class TestParseFrequency:
    """Test frequency parsing."""
    
    def test_simple_units(self):
        """Test parsing simple frequency units."""
        assert parse_frequency("H") == (1, "H")
        assert parse_frequency("D") == (1, "D")
        assert parse_frequency("M") == (1, "M")
        assert parse_frequency("Y") == (1, "Y")
    
    def test_with_multipliers(self):
        """Test parsing frequencies with multipliers."""
        assert parse_frequency("15T") == (15, "T")
        assert parse_frequency("5min") == (5, "T")
        assert parse_frequency("30T") == (30, "T")
    
    def test_aliases(self):
        """Test various pandas frequency aliases."""
        assert parse_frequency("min")[1] == "T"
        assert parse_frequency("h")[1] == "H"
        assert parse_frequency("YE")[1] == "Y"


class TestGetPatchSize:
    """Test patch size computation."""
    
    def test_yearly_quarterly(self):
        """Test patch sizes for low frequency data."""
        assert get_patch_size_for_freq("Y") == 8
        assert get_patch_size_for_freq("Q") == 16
    
    def test_monthly_weekly(self):
        """Test patch sizes for medium frequency data."""
        assert get_patch_size_for_freq("M") == 32
        assert get_patch_size_for_freq("W") == 64
    
    def test_daily_hourly(self):
        """Test patch sizes for daily and hourly data."""
        assert get_patch_size_for_freq("D") == 128
        assert get_patch_size_for_freq("H") == 128
    
    def test_minute_second(self):
        """Test patch sizes for high frequency data."""
        assert get_patch_size_for_freq("T") == 256
        assert get_patch_size_for_freq("15T") == 256
        assert get_patch_size_for_freq("5min") == 256
        assert get_patch_size_for_freq("S") == 256

