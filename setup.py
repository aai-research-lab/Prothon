from setuptools import setup, find_packages

setup(
    name="Prothon",
    version="2.0.0",
    author="Adekunle aina",
    author_email="kunleaina@gmail.com",
    description="Efficient comparison of protein ensembles using local order parameters",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "mdtraj",
        "scipy",
        "matplotlib"
    ],
    entry_points={
        "console_scripts": [
            "prothon=Prothon.cli:main"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ]
)

