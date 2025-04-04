from setuptools import setup, find_packages

setup(
    name="nlp-project",
    version="0.1.0",
    description="NLP project with LSTM",
    author="Magesh",
    author_email="mageshytdev@gmail.com",
    packages=find_packages(),
    package_dir={"": "./"},
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "tensorflow",
        "nltk",
        "matplotlib",
        "seaborn",
    ],
    # Adding extras_require for optional dependencies
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
            "jupyter",
        ],
    },
    # Adding entry_points for command-line scripts
    entry_points={
        # TODO: Add a commad-line script
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
