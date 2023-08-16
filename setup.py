import pathlib
import setuptools

# The directory containing this file
TOPLEVEL_DIR = pathlib.Path(__file__).parent.absolute()
ABOUT_FILE = TOPLEVEL_DIR / "pySC" / "__init__.py"
README = TOPLEVEL_DIR / "README.md"

# Information about pySC package
ABOUT_PYSC: dict = {}
with ABOUT_FILE.open("r") as f:
    exec(f.read(), ABOUT_PYSC)

with README.open("r") as docs:
    long_description = docs.read()

# Dependencies for the package itself
DEPENDENCIES = [
    "numpy>=1.22.0",
    "scipy>=1.8.0"
    "matplotlib>=3.5.0",
    "accelerator-toolbox>=0.4.0"

]

# Extra dependencies
EXTRA_DEPENDENCIES = {
    "test": [
        "pytest>=7.0",
        "pytest-cov>=3.0",


    ],
    "doc": [
        "sphinx",
        "travis-sphinx",
        "sphinx-rtd-theme"
    ]}

EXTRA_DEPENDENCIES.update(
    {'all': [elem for list_ in EXTRA_DEPENDENCIES.values() for elem in list_]}
)

setuptools.setup(
    name=ABOUT_PYSC["__title__"],
    version=ABOUT_PYSC["__version__"],
    description=ABOUT_PYSC["__description__"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=ABOUT_PYSC["__author__"],
    author_email=ABOUT_PYSC["__author_email__"],
    url=ABOUT_PYSC["__url__"],
    packages=setuptools.find_packages(exclude=["tests", "doc"]),
    python_requires=">=3.8",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    install_requires=DEPENDENCIES,
    tests_require=EXTRA_DEPENDENCIES["test"],
    extras_require=EXTRA_DEPENDENCIES,
)
