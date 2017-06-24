"""A setuptools based setup module.
Template taken from: https://raw.githubusercontent.com/pypa/sampleproject/master/setup.py
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='rnr-debugging-scripts',
    version='0.1.3',

    description='A Project with Retrieve and Rank helper scripts and examples',
    long_description=long_description,

    url='https://github.ibm.com/rchakravarti/rnr-debugging-scripts',
    author='rchakravarti',
    author_email='rchakravarti@us.ibm.com',

    license='Apache 2.0',

    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Bluemix :: Retrieve and Rank',
        'License :: OSI Approved :: Apache License',
        'Programming Language :: Python :: 3.5',
    ],

    keywords='retrieve and rank RnR discovery bluemix evaluation information retrieval sample example debugging scripts',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=[line for line in open('requirements.txt').read().splitlines() if not line.startswith('--')],

    # for example:
    # $ pip install -e .[examples]
    extras_require={
        'examples': [line for line in open('requirements-examples.txt').read().splitlines() if not line.startswith('--')],
    },
)
