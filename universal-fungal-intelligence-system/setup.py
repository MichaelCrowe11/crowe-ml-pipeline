from setuptools import setup, find_packages

setup(
    name='universal-fungal-intelligence-system',
    version='0.1.0',
    author='Michael Crowe',
    author_email='michaelcrowe11@example.com',
    description='A comprehensive system for analyzing fungal species and their chemical potential for human therapeutics.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/universal-fungal-intelligence-system',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'pandas',
        'rdkit',
        'scikit-learn',
        'torch',
        'biopython',
        'aiohttp',
        'requests',
    ],
)