"""
__init__.py

This package installer exposes the top-level training entry point for the Seemore Vision-Language Model
that was expanded / optimized by Lauren Rutledge for the sake of the seemore interview assignment.

Author: Lauren Rutledge
Date: April 2025
"""


# Making the train_model function available at the package level: 
from training.train_model import train_model

__all__ = ["train_model"]
