# coding: utf-8
# Copyright (c) Tingzheng Hou and Lu Jiang.
# Distributed under the terms of the MIT License.

"""
This module implements a core class FeatureMaker for generating
lammps data files and input files from xsf files, running lammps
and parsing the lammps outputs to training data file.

"""

import numpy as np
from ase.io.xsf import read_xsf
from ase.io.lammpsdata import write_lammps_data
from tqdm import tqdm
from lammps import lammps
from pymatgen.io.lammps.inputs import write_lammps_inputs
from scipy.io import savemat
import os
import re
import shutil
import sys
import warnings

__author__ = "Tingzheng Hou and Lu Jiang"
__copyright__ = "Copyright 2020, Tingzheng Hou and Lu Jiang"
__version__ = "1.0"
__maintainer__ = "Lu Jiang"
__email__ = "lu_jiang@berkeley.edu"
__date__ = "May 3, 2020"

TEMPLATE_STRUCTURE = '''log              ${filename}.log

units			metal
boundary 		p p p
atom_style 		atomic 

read_data 	      ${filename}
#pair_style           snap
pair_style           eam/fs
pair_coeff           * * Zr_3.eam.fs Zr


compute          b all sna/atom ${rcutfac} ${rfac0} ${twojmax} ${R_1} 1.0
compute          bsum all reduce sum c_b[*]


thermo           1
thermo_style     custom c_bsum[*]

timestep            0.001

run               0'''

TEMPLATE_ATOM = '''log              ${filename}.log

units			metal
boundary 		p p p
atom_style 		atomic 

read_data 	      ${filename}
#pair_style           snap
pair_style           eam/fs
pair_coeff           * * Zr_3.eam.fs Zr


compute          b all sna/atom ${rcutfac} ${rfac0} ${twojmax} ${R_1} 1.0
dump             1 all custom 1 ${filename}.dump c_b[*] 

timestep            0.001

run               0'''

XSF_DIR = "/Users/th/Downloads/xsffiles/"

DATA_DIR = '/Users/th/Downloads/datafiles/'


class FeatureMaker:

    def __init__(self, xsf_dir, data_dir, sna_setting, mode="structure", screen=None):
        """
        Base constructor.

        Args:
            xsf_dir (str): directory to the xsf files.
            data_dir (str): directory to the data files.
            mode (str): training mode.
            screen (bool):

        """
        print("Start making features for training Re potential")
        self.xsf_dir = xsf_dir
        self.data_dir = data_dir
        self.log_dir = data_dir
        self.run_dir = data_dir
        self.sna_setting = sna_setting
        self.screen = screen
        self.generate_data()
        if mode == "structure":
            self.mode = mode
            self.batch_run(template=TEMPLATE_STRUCTURE)
            self.training_n = self.read_n()
            self.training_samples = self.read_logs(self.training_n, mode="structure")
            self.training_energy = self.read_energy(self.training_n, mode="structure")
            print("Made structure-wise features with:")
            print(self.training_samples.shape, "training samples")
            print(self.training_energy.shape, "labels")
        elif mode == "atom":
            self.mode = mode
            self.batch_run(template=TEMPLATE_ATOM)
            self.training_n = self.read_n()
            self.training_samples = self.read_logs(self.training_n, mode="atom")
            self.training_energy = self.read_energy(self.training_n, mode="atom")
            print("Made atom-wise features with:")
            print(self.training_samples.shape, "training samples")
            print(self.training_energy.shape, "labels")
        else:
            raise ValueError("Mode not supported.")

    def generate_data(self, make_dir_if_not_present=True):
        """
        Generate lammps data files from the available .xsf structure files.

        Args:

            make_dir_if_not_present (bool): Whether to create the datafile
                output directory when not exist.

        """
        print('Generating lammps data files...')
        if make_dir_if_not_present and not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        shutil.copyfile(os.path.join(self.xsf_dir, 'Zr_3.eam.fs'),
                        os.path.join(self.data_dir, 'Zr_3.eam.fs'))
        for i in tqdm(range(6500)):
            filename = "vasprun" + str((i + 1)) + ".xsf"
            with open(os.path.join(self.xsf_dir, filename), 'r') as f:
                try:
                    vasprun = read_xsf(f)
                    write_lammps_data(self.data_dir + filename[:-4] + ".data", vasprun)
                except UnicodeDecodeError:
                    print(filename)

    @staticmethod
    def run_lammps(inputname, template, settings=None, screen=None):
        """
        Run one lammps job.

        Args:
            inputname (str): the name of lammps input file name.
            template (str): lammps input template.
            settings (dict): Contains values to be written to the
                placeholders, e.g., {'temperature': 1}. Default to None.
            screen (bool): Whether to print lammps logs on screen.
                Default to None.

        """
        write_lammps_inputs('.', template, settings=settings, script_filename=inputname)
        if not screen:
            lmp = lammps(cmdargs=["-screen", "none"])
        else:
            lmp = lammps()
        lmp.file(inputname)

    def batch_run(self, template=TEMPLATE_STRUCTURE):
        """
        Run batch lammps jobs.

        Args:
            template (str): lammps input template.

        """
        warnings.filterwarnings("ignore")
        print('\n')
        print('Running lammps...')
        os.chdir(self.run_dir)
        for i in tqdm(range(6500)):
            dataname = "vasprun" + str((i + 1)) + ".data"
            inputname = "vasprun" + str((i + 1)) + ".input"
            settings = {'filename': dataname}
            settings.update(self.sna_setting)
            self.run_lammps(inputname, template, settings=settings, screen=self.screen)

    @staticmethod
    def read_log(log_dir, filename, n_i, mode="structure"):
        """
        Read one lammps log file.

        Args:
            log_dir (str): the directory of the log file.
            filename (str): the name of the log file.
            n_i (numpy.int): the first feature which is the
                number of atoms in the structure.
            mode (str): specify the output is whether atom-wise
                feature or structure-wise feature.

        Returns:
            A numpy array of the bispectrum terms

        """
        if mode == "structure":
            begin_flag = "c_bsum[1] c_bsum[2] c_bsum[3]"
            try:
                begin_i = None
                with open(os.path.join(log_dir, filename), "r") as f:
                    lines = f.readlines()
                    for i, l in enumerate(lines):
                        if l.startswith(begin_flag):
                            begin_i = i + 1
                            break
                    line = lines[begin_i].split()
            except FileNotFoundError or TypeError:
                print('Unable to read log')
                sys.exit()
            array = [float(n_i)]
            for num in line:
                array.append(float(num))
            array = np.array(array)
            return array
        elif mode == "atom":
            try:
                with open(os.path.join(log_dir, filename), "r") as f:
                    lines = f.readlines()
                    atom_lines = lines[9:]
            except FileNotFoundError or TypeError:
                print('Unable to read dump')
                sys.exit()
            arrays = []
            for i, line in enumerate(atom_lines):
                line = line.split()
                array = [float(1)]
                for num in line:
                    array.append(float(num))
                array = np.array(array)
                arrays.append(array)
            return np.array(arrays)
        else:
            raise ValueError("Mode not supported.")

    def read_logs(self, n, mode="structure"):
        """
        Read batch lammps log files.

        Args:
            n (numpy.array): the first feature which is the
                number of atoms in the structure.
            mode (str): specify the output is whether atom-wise
                feature or structure-wise feature.

        Returns:
            A numpy ndarray of the bispectrum terms

        """
        print('\n')
        print('Reading lammps logs...')
        samples = []
        if mode == "structure":
            for i in tqdm(range(6500)):
                logname = "vasprun" + str((i + 1)) + ".data.log"
                sample = self.read_log(self.log_dir, logname, n[i], mode)
                samples.append(sample)
            samples = np.array(samples)
            return samples
        elif mode == "atom":
            for i in tqdm(range(6500)):
                logname = "vasprun" + str((i + 1)) + ".data.dump"
                sample = self.read_log(self.log_dir, logname, n[i], mode)
                samples.append(sample)
            samples = np.concatenate(tuple(samples), axis=0)
            return samples
        else:
            raise ValueError("Mode not supported.")

    def read_energy(self, n, mode="structure"):
        """
        Read energy labels from xsf files.

        Args:
            n (numpy.array): number of atoms for each structure.
            mode (str): specify the output is whether atom-wise
                feature or structure-wise feature.

        Returns:
            A numpy array of the energy

        """
        print('\n')
        print('Reading energy...')
        energy = []
        if mode == "structure":
            for i in tqdm(range(6500)):
                filename = "vasprun" + str((i + 1)) + ".xsf"
                with open(os.path.join(self.xsf_dir, filename), 'r') as f:
                    line = f.readlines()[0]
                    energy_i = float(re.split('\s+', line)[-3])
                    energy.append(energy_i)
            energy = np.array(energy)
            return energy.reshape(-1, 1)
        elif mode == "atom":
            for i in tqdm(range(6500)):
                filename = "vasprun" + str((i + 1)) + ".xsf"
                with open(os.path.join(self.xsf_dir, filename), 'r') as f:
                    line = f.readlines()[0]
                    num_atom = int(n[i])
                    energy_i = float(re.split('\s+', line)[-3]) / num_atom
                    for _ in range(num_atom):
                        energy.append(energy_i)
            energy = np.array(energy)
            return energy.reshape(-1, 1)
        else:
            raise ValueError("Mode not supported.")

    def read_n(self):
        """
        Read number of atoms from .xsf structure files.

        Returns:
            A numpy array of number of atoms

        """
        print('\n')
        print('Reading numbers of atoms...')
        num_array = []
        for i in tqdm(range(6500)):
            dataname = "vasprun" + str((i + 1)) + ".data"
            with open(os.path.join(self.data_dir, dataname), 'r') as f:
                line = f.readlines()[2]
                count = re.split('\s+', line)[0]
                num_array.append(int(count))
        num_array = np.array(num_array)
        return num_array.reshape(-1, 1)

    def save_samples(self, output_dir, filename):
        """
        Save samples to file for potential training

        Args:
            output_dir (str): the saving directory.
            filename (str): the name of the output file.

        """
        print('\n')
        print('Saving sample to file...')
        savemat(os.path.join(output_dir, filename),
                {'X': self.training_samples, 'y': self.training_energy})
        print('\n')
        print('Sample file ' + filename + ' saved.')


# fm = FeatureMaker(XSF_DIR, DATA_DIR,
#                   {"rcutfac": 1.4, "rfac0": 0.95, "twojmax": 6, "R_1": 2.0},
#                   mode="structure", screen=False)
# fm.save_samples(DATA_DIR, "re.mat")
