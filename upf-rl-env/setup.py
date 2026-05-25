from setuptools import find_packages, setup

setup(
    name="upf-rl-env",
    version="0.1.1",
    description="Gymnasium env + DQN agent for slice-aware MBR control on BESS-UPF",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "gymnasium>=0.29",
        "numpy>=1.24",
        "torch>=2.0",
        "tensorboard>=2.14",
        "scipy>=1.10",
    ],
)
