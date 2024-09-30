import setuptools


setuptools.setup(
    name="bb-energy-prediction",
    version="0.1",
    description="bb energy prediction",
    packages=setuptools.find_packages(),
    install_requires=[
        "torch>=1.13.0+cu117,<2.3.0",
        "optuna>=3.1.0,<3.2.0",
        "optuna-dashboard>=0.8.0,<0.9.0",
        "pandas>=1.5.0,<1.6.0",
        "numpy>=1.23.0,<1.24.0",
        "tqdm>=4.64.0,<4.65.0",
        "scikit-learn>=1.1.0,<1.2.0",
        "python-dotenv>=0.21.0,<0.22.0",
        "torchaudio",
        "torchvision",
        "torchtext",
    ],
    dependency_links=[
        "https://download.pytorch.org/whl/cu117",
    ],
    python_requires=">=3.10",
)
