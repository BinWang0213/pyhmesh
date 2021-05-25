from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# https://github.com/pybind/python_example/
is_called = [
    "_FastHJ"
]

files = [
    "FastHJ.cpp"
]

# no CGAL libraries necessary from CGAL 5.0 onwards
ext_modules = [
    Pybind11Extension(loc, [fi])
    for fi, loc in zip(files, is_called)
]

if __name__ == "__main__":
    setup(
        name='fastHJ',
        version='1.0.0',
        cmdclass={"build_ext": build_ext},
        ext_modules=ext_modules,
        zip_safe=False,
    )