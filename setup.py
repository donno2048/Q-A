from setuptools import setup, find_packages
setup(
    name = 'WikiQA',
    version = '1.0.1',
    description = 'A package to process wikipedia questions',
    long_description = open('README.md').read(),
    long_description_content_type = "text/markdown",
    packages = find_packages(),
    url = 'https://github.com/donno2048/Q-A',
    license = 'MIT',
    author = 'Elisha Hollander',
    classifiers = ['Operating System :: Microsoft :: Windows','License :: OSI Approved :: MIT License','Programming Language :: Python :: 3'],
    install_requires = ['argparse', 'prettytable', 'regex', 'sklearn', 'spacy', 'termcolor', 'torch==1.7.0+cpu', 'wexpect']
)
