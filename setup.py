# Shim so `python setup.py sdist` keeps working (used by the generated Vertex AI submit scripts).
# All package metadata lives in pyproject.toml.
from setuptools import setup

setup()
