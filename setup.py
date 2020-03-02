import io
import os
import sys
from setuptools import find_packages, setup

# Package meta-data.
NAME = 'fennlp'
DESCRIPTION = 'An out-of-the-box NLP toolkit can easily help you solve tasks such ' \
              'as entity recognition, relationship extraction, text classfication and so on.'
URL = 'https://github.com/kyzhouhzau/FenNLP'
EMAIL = 'zhoukaiyinhzau@gmail.com'
AUTHOR = 'Kaiyin Zhou'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.0.0'

REQUIRED = [
    # 'tensorflow>=2.0',
    'typeguard',
    'gensim'
]

# What packages are optional?
EXTRAS = {
    'fancy feature': ['numpy'],
}

# Where the magic happens:
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='Apache',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: Apache License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
)