from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension("mcts", ["multimcts/mcts.c"]),
]

setup(
    name="multimcts",
    ext_modules=cythonize(extensions),
)
