"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))


setup(
    name='torchinceptionresnetv2',  # Required
    version='0.0.1',  # Required
    description='PyTorch implementation of the neural network introduced by Szegedy et. al in '
                '"Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning"',  # Optional
    long_description_content_type='text/markdown',  # Optional (see note above)
    url='https://github.com/mhconradt/InceptionResNetV2',  # Optional
    author='Maxwell Conradt',  # Optional
    author_email='mhconradt@protonmail.com',  # Optional
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        # Pick your license as you wish
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        # These classifiers are *not* checked by 'pip install'. See instead
        # 'python_requires' below.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    keywords='neural networks machine learning inception resnet',  # Optional
    packages=find_packages(),  # Required
    python_requires='>=3, <4',
    install_requires=['torch'],  # Optional
)
