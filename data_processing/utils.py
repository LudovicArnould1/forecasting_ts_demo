"""Utility functions for time series data processing."""

import re
from typing import Literal


def parse_frequency(freq_str: str) -> tuple[int, Literal["Y", "Q", "M", "W", "D", "H", "T", "S"]]:
    """Parse pandas frequency string to (multiplier, base_unit).
    
    Args:
        freq_str: Frequency string like 'H', '15T', '5min', 'D', etc.
        
    Returns:
        Tuple of (multiplier, base_unit) where base_unit is one of:
        - 'Y': Year
        - 'Q': Quarter
        - 'M': Month
        - 'W': Week
        - 'D': Day
        - 'H': Hour
        - 'T': Minute (T for Time)
        - 'S': Second
        
    Examples:
        >>> parse_frequency('H')
        (1, 'H')
        >>> parse_frequency('15T')
        (15, 'T')
        >>> parse_frequency('5min')
        (5, 'T')
    """
    # Common pandas frequency aliases
    unit_map = {
        "Y": "Y", "YE": "Y", "YS": "Y", "A": "Y", "AS": "Y",  # Year
        "Q": "Q", "QE": "Q", "QS": "Q",  # Quarter
        "M": "M", "ME": "M", "MS": "M",  # Month
        "W": "W",  # Week
        "D": "D", "B": "D", "C": "D",  # Day (business day, custom)
        "H": "H", "h": "H",  # Hour
        "T": "T", "min": "T",  # Minute
        "S": "S", "s": "S",  # Second
    }
    
    # Try to match pattern: optional number + unit
    match = re.match(r"^(\d*)([A-Za-z]+)$", freq_str)
    if not match:
        raise ValueError(f"Cannot parse frequency string: {freq_str}")
    
    multiplier_str, unit = match.groups()
    multiplier = int(multiplier_str) if multiplier_str else 1
    
    # Normalize unit
    base_unit = unit_map.get(unit)
    if base_unit is None:
        raise ValueError(f"Unknown frequency unit: {unit}")
    
    return multiplier, base_unit


def get_patch_size_for_freq(freq_str: str) -> int:
    """Get patch size based on frequency as per Moirai paper.
    
    The patch size determines how many consecutive timesteps are grouped together.
    Lower frequency (yearly) -> smaller patches
    Higher frequency (minute) -> larger patches
    
    Args:
        freq_str: Frequency string like 'H', '15T', 'D', etc.
        
    Returns:
        Patch size (number of timesteps per patch)
        
    Examples:
        >>> get_patch_size_for_freq('Y')
        8
        >>> get_patch_size_for_freq('H')
        128
        >>> get_patch_size_for_freq('15T')
        128
    """
    _, base_unit = parse_frequency(freq_str)
    
    # Based on Moirai paper's patch size mapping
    patch_size_map = {
        "Y": 8,      # Yearly
        "Q": 16,     # Quarterly
        "M": 32,     # Monthly
        "W": 64,     # Weekly
        "D": 128,    # Daily
        "H": 128,    # Hourly
        "T": 256,    # Minute
        "S": 256,    # Second
    }
    
    return patch_size_map[base_unit]

