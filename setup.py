from setuptools import setup, find_packages

from readparse import __version__


setup(
    name='readparse',
    version=__version__,

    url='https://github.com/apt-get-nat/PARSE',
    author='Nat Mathews',
    author_email='n.h.mathews@nasa.gov',

    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'astropy',
        'streamtracer'
    ]
)
