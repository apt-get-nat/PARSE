from setuptools import setup

from parse import __version__


setup(
    name='parse',
    version=__version__,

    url='https://github.com/apt-get-nat/PARSE',
    author='Nat Mathews',
    author_email='n.h.mathews@nasa.gov',

    py_modules=['parse'],
    install_requires=[
        'numpy',
        'scipy',
        'astropy',
        'streamtracer'
    ]
)
