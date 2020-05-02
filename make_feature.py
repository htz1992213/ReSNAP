from ase.io.xsf import read_xsf
from ase.io.lammpsdata import write_lammps_data
from tqdm import tqdm
from lammps import lammps
import numpy as np
from pymatgen.io.lammps.inputs import write_lammps_inputs
from scipy.io import savemat
import os
import re
import shutil
import sys
import warnings

TEMPLATE_STRUCTURE = '''log              ${filename}.log

units			metal
boundary 		p p p
atom_style 		atomic 

read_data 	      ${filename}
#pair_style           snap
pair_style           eam/fs
pair_coeff           * * Zr_3.eam.fs Zr


compute          b all sna/atom 1.4 0.95 6 2.0 1.0
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


compute          b all sna/atom 1.4 0.95 6 2.0 1.0
dump             1 all custom 1 ${filename}.dump c_b[*] 

timestep            0.001

run               0'''

XSF_DIR = "/Users/th/Downloads/xsffiles"
DATA_DIR = '/Users/th/Downloads/datafiles/'


def generate_data(xsf_dir, data_dir, make_dir_if_not_present=True):
    """
    Generate lammps data files from the available .xsf structure files.

    Args:
        xsf_dir (str): directory to the xsf files.
        data_dir (str): output directory to the data files.
        make_dir_if_not_present (bool): Whether to create the datafile
            output directory when not exist.

    """
    print('generating lammps data files...')
    if make_dir_if_not_present and not os.path.exists(data_dir):
        os.makedirs(data_dir)
    shutil.copyfile(os.path.join(xsf_dir, 'Zr_3.eam.fs'),
                    os.path.join(data_dir, 'Zr_3.eam.fs'))
    for i in tqdm(range(6500)):
        filename = "vasprun" + str((i + 1)) + ".xsf"
        with open(os.path.join(xsf_dir, filename), 'r') as f:
            try:
                vasprun = read_xsf(f)
                write_lammps_data(data_dir + filename[:-4] + ".data", vasprun)
            except UnicodeDecodeError:
                print(filename)


def run_lammps(inputname, dataname, template=TEMPLATE_STRUCTURE):
    """
    Run one lammps job.

    Args:
        inputname (str): the name of lammps input file name.
        dataname (str): the name of lammps data file name.
        template (str): lammps input template.

    """
    write_lammps_inputs('.', template, settings={'filename': dataname},
                        script_filename=inputname)
    lmp = lammps(cmdargs=["-screen", "none"])
    lmp.file(inputname)


def batch_run(run_dir, template=TEMPLATE_STRUCTURE):
    """
    Run batch lammps jobs.

    Args:
        run_dir (str): the working directory of lammps jobs.
        template (str): lammps input template.

    """
    warnings.filterwarnings("ignore")
    print('\n')
    print('running lammps...')
    os.chdir(run_dir)
    for i in tqdm(range(6500)):
        dataname = "vasprun" + str((i + 1)) + ".data"
        inputname = "vasprun" + str((i + 1)) + ".input"
        run_lammps(inputname, dataname, template)


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
        except:
            print('Unable to read log')
            sys.exit()
        line = lines[begin_i].split()
        array = [float(n_i)]
        for num in line:
            array.append(float(num))
        array = np.array(array)
        return array
    elif mode == "atom":
        try:
            with open(os.path.join(log_dir, filename), "r") as f:
                lines = f.readlines()
        except:
            print('Unable to read dump')
            sys.exit()
        arrays = []
        atom_lines = lines[9:]
        for i, line in enumerate(atom_lines):
            line = line.split()
            array = [float(1)]
            for num in line:
                array.append(float(num))
            array = np.array(array)
            arrays.append(array)
        return np.array(arrays)
    else:
        print("unsupported mode")
        return None


def read_logs(log_dir, n, mode="structure"):
    """
    Read batch lammps log files.

    Args:
        log_dir (str): the directory to the log files.
        n (numpy.array): the first feature which is the
            number of atoms in the structure.
        mode (str): specify the output is whether atom-wise
            feature or structure-wise feature.

    Returns:
        A numpy ndarray of the bispectrum terms

    """
    print('\n')
    print('reading lammps logs...')
    samples = []
    if mode == "structure":
        for i in tqdm(range(6500)):
            logname = "vasprun" + str((i + 1)) + ".data.log"
            sample = read_log(log_dir, logname, n[i], mode)
            samples.append(sample)
        samples = np.array(samples)
        return samples
    elif mode == "atom":
        for i in tqdm(range(6500)):
            logname = "vasprun" + str((i + 1)) + ".data.dump"
            sample = read_log(log_dir, logname, n[i], mode)
            samples.append(sample)
        samples = np.concatenate(tuple(samples), axis=0)
        return samples
    else:
        print("unsupported mode")
        return None


def read_energy(xsf_dir, n, mode="structure"):
    """
    Read energy labels from xsf files.

    Args:
        xsf_dir (str): the directory to the xsf files.
        n (numpy.array): number of atoms for each structure.
        mode (str): specify the output is whether atom-wise
            feature or structure-wise feature.

    Returns:
        A numpy array of the energy

    """
    print('\n')
    print('reading energy...')
    energy = []
    if mode == "structure":
        for i in tqdm(range(6500)):
            filename = "vasprun" + str((i + 1)) + ".xsf"
            with open(os.path.join(xsf_dir, filename), 'r') as f:
                line = f.readlines()[0]
                energy_i = float(re.split('\s+', line)[-3])
                energy.append(energy_i)
        energy = np.array(energy)
        return energy.reshape(-1, 1)
    elif mode == "atom":
        for i in tqdm(range(6500)):
            filename = "vasprun" + str((i + 1)) + ".xsf"
            with open(os.path.join(xsf_dir, filename), 'r') as f:
                line = f.readlines()[0]
                num_atom = int(n[i])
                energy_i = float(re.split('\s+', line)[-3]) / num_atom
                for _ in range(num_atom):
                    energy.append(energy_i)
        energy = np.array(energy)
        return energy.reshape(-1, 1)
    else:
        print("unsupported mode")
        return None


def read_n(data_dir):
    """
    Read number of atoms from .xsf structure files.

    Args:
        data_dir (str): the directory to the data files.

    Returns:
        A numpy array of number of atoms

    """
    print('\n')
    print('reading numbers of atoms...')
    num_array = []
    for i in tqdm(range(6500)):
        dataname = "vasprun" + str((i + 1)) + ".data"
        with open(os.path.join(data_dir, dataname), 'r') as f:
            line = f.readlines()[2]
            count = re.split('\s+', line)[0]
            num_array.append(int(count))
    num_array = np.array(num_array)
    return num_array.reshape(-1, 1)


def save_samples(samples, energy, output_dir, filename):
    """
    Save samples to file for potential training

    Args:
        samples (numpy.array): bispectrum terms.
        energy (numpy.array): energy labels.
        output_dir (str): the saving directory.
        filename (str): the name of the output file.

    """
    print('\n')
    print('saving sample to file...')
    savemat(os.path.join(output_dir, filename), {'X': samples, 'y': energy})
    print('\n')
    print('sample file ' + filename + ' saved.')


def structure_feature():
    # generate_data(XSF_DIR, DATA_DIR)
    # batch_run(DATA_DIR)
    training_n = read_n(DATA_DIR)
    training_samples = read_logs(DATA_DIR, training_n)
    training_energy = read_energy(XSF_DIR, training_n)
    print("Made structure-wise features with:")
    print(training_samples.shape, "training samples")
    print(training_energy.shape, "labels")
    save_samples(training_samples, training_energy, DATA_DIR, 're.mat')


def atom_feature():
    # generate_data(XSF_DIR, DATA_DIR)
    # batch_run(DATA_DIR, template=TEMPLATE_ATOM)
    training_n = read_n(DATA_DIR)
    training_samples = read_logs(DATA_DIR, training_n, mode="atom")
    training_energy = read_energy(XSF_DIR, training_n, mode="atom")
    print("Made atom-wise features with:")
    print(training_samples.shape, "training samples")
    print(training_energy.shape, "labels")
    save_samples(training_samples, training_energy, DATA_DIR, 're_atom.mat')


structure_feature()
