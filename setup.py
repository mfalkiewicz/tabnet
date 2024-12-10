from setuptools import setup, find_packages

# Import versioneer only after it's installed
try:
    import versioneer

    version = versioneer.get_version()
    cmdclass = versioneer.get_cmdclass()
except ImportError:
    version = "0.0.0"
    cmdclass = {}

setup(
    name="pytorch_tabnet",
    version=version,
    cmdclass=cmdclass,
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
        "numpy>=1.19.0,<2.0.0",
        "scipy>=1.5.0",
        "pandas>=1.0.0",
        "scikit-learn>=0.24.0",
        "pyspark>=3.0.0",
        "petastorm>=0.11.0",
        "pyarrow<12.0.0",
    ],
    python_requires=">=3.7",
)
