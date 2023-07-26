from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

setup(
    name='multimcts',
    version='0.4',
    description='Monte Carlo Tree Search for multiple teams',
    author='Taylor Vance',
    author_email='mirrors.cities0w@icloud.com',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license='MIT',
    url='https://github.com/taylorvance/multimcts',
    packages=find_packages(),
    ext_modules=cythonize([
        Extension('multimcts.mcts', ['multimcts/mcts.pyx']),
    ]),
    package_data={'multimcts': ['*.pyx']},
)
