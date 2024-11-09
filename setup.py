from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "trie",
        ["trie-encoder.cpp"],
        cxx_std=17,
    ),
]

setup(
    name="trie",
    version="0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
