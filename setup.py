from setuptools import setup, find_packages, Extension

setup(
    name='boxmodel',
    version='0.0.1',
    python_requires=">=3.6",
    install_requires=[
    ],
    ext_modules = [
        Extension(
            name = 'greet', 
            sources =['boxmodel/solvers/rk.c']
        ),
    ],
    packages=find_packages(
        where='boxmodel',
        include=['boxmodel']
    ),
)