# setup.py
from setuptools import setup,find_packages

setup(
    name='TabulaRL',
    packages=[package for package in find_packages()
                if package.startswith('TabulaRL')],
    version='0.1.0',
)