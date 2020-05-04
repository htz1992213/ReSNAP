import os

from setuptools import setup, find_packages

module_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    setup(
        name='re_ml',
        version='2020.5.3',
        packages=find_packages(),
        install_requires=["ase", "tqdm", "lammps", "matplotlib", "scikit-learn",
                          "numpy", "scipy", "pymatgen"],
        description='Repository for training Re potential',
        python_requires='>=3.6'
    )
