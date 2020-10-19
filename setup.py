from setuptools import setup, find_packages

packages = find_packages()

setup(
    name="MIMIC",
    version="9999",
    description="Multimodal implementation of a VAE for the MIMIC-CXR Database",
    author="Hendrik Klug",
    author_email="klugh@ethz.ch",
    url="https://github.com/Jimmy2027/MIMIC",
    keywords=["MMVAE", "data analysis", "deep learning"],
    classifiers=[],
    install_requires=[],
    provides=["mimic"],
    packages=packages,
    include_package_data=True,
    extras_require={},
)