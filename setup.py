from setuptools import setup, find_packages, Extension

setup(
    ext_modules = [
        Extension(
            name = 'numerics', 
            sources =['src/numerics/numerics.cpp']
        ),
    ],
)