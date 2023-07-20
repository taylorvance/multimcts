from setuptools import setup
from Cython.Build import cythonize

setup(
    name="multimcts",
    version="0.2",
    author="Taylor Vance",
    author_email="mirrors.cities0w@icloud.com",
    url="https://github.com/taylorvance/multimcts",
    description="Monte Carlo Tree Search for multiple teams",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    packages=["multimcts"],
    ext_modules=cythonize("multimcts/mcts.pyx"),
    zip_safe=False,
)
