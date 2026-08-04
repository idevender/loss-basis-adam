from setuptools import find_packages, setup

setup(
    name="flowadam",
    version="0.2.0",
    description="FlowAdam optimizer and the gauge-equivariance implicit-bias experiments",
    author="Devender Singh",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "torchdiffeq>=0.2",
    ],
)
