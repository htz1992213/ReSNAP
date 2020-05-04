import os

from setuptools import setup, find_packages

module_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    setup(
        name='resnap',
        version='2020.5.4',
        packages=find_packages(),
        install_requires=["tqdm", "pymatgen", "matplotlib", "scikit-learn",
                          "numpy", "scipy", "monty"],
        optional_requires={"make_feature": ["ase", "lammps"]},
        description='Repository for training Re potential',
        python_requires='>=3.6'
    )
