import math
import numpy as np

"""
This module implements a core class Structure for lammps data file
i/o and adding strain to structure.


lna has the form: [a,b,c,alpha,beta,gamma].
The lattice vectors have the form of [[x,0,0],[xy,y,0],[xz,yz,z]],
and the cartesian coordinates is arranged accordingly.
"""

__author__ = "Lu Jiang and Tingzheng Hou"
__copyright__ = "Copyright 2020, Tingzheng Hou and Lu Jiang"
__version__ = "1.0"
__maintainer__ = "Lu Jiang"
__email__ = "lu_jiang@berkeley.edu"
__date__ = "May 3, 2020"


class Structure:
    def __init__(self, filename, filetype="LAMMPSdat"):

        ftype = filetype
        fname = filename
        data = []
        with open(fname, "r") as f:
            for line in f.readlines():
                data += [line.split()]

        if ftype == "LAMMPSdat":
            for line in data:
                if 'atoms' in line:
                    self.natoms = int(line[0])
            cartesianc0 = []
            lattice0 = []
            for i in range(len(data)):
                if 'Atoms' in data[i]:
                    for j in range(self.natoms):
                        cartesianc0 += [[float(data[i+2+j][2]),
                                         float(data[i+2+j][3]),
                                         float(data[i+2+j][4])]]
                    
                if 'types' in data[i]:     
                    lattice0 += [[float(data[i+1][1]) - float(data[i+1][0]),
                                  0., 0.]]
                    lattice0 += [[0., float(data[i+2][1]) - float(data[i+2][0]),
                                  0.]]
                    lattice0 += [[0., 0.,
                                  float(data[i+3][1]) - float(data[i+3][0])]]
                if 'xy' in data[i]:
                    lattice0[0][1] = float(data[i][0])
                    lattice0[0][2] = float(data[i][1])
                    lattice0[1][2] = float(data[i][2])
            self.cartesianc = cartesianc0
            self.lattice = lattice0
            self.directc = self.get_directc()
            self.lna = self.get_lna()
        else:
            raise ValueError("Unsupported file format.")
                
    def update_cartsc(self):
        self.cartesianc = []
        for i in range(self.natoms):
            self.cartesianc += [np.dot(self.directc[i], self.lattice)]

    def get_directc(self):
        dire = []
        for i in range(self.natoms):
            dire += [np.dot(self.cartesianc[i], np.linalg.inv(self.lattice))]
        return dire

    def get_lna(self):
        a = math.sqrt(np.dot(self.lattice[0], self.lattice[0]))
        b = math.sqrt(np.dot(self.lattice[1], self.lattice[1]))
        c = math.sqrt(np.dot(self.lattice[2], self.lattice[2]))
        alpha = np.arccos(np.dot(self.lattice[1],
                                 self.lattice[2])/(b*c)) * 180 / np.pi
        beta = np.arccos(np.dot(self.lattice[0],
                                self.lattice[2])/(a*c)) * 180 / np.pi
        gamma = np.arccos(np.dot(self.lattice[0],
                                 self.lattice[1])/(a*b)) * 180 / np.pi
        return [a, b, c, alpha, beta, gamma]  # angle in degrees

    def get_lattice(self, lna):
        [a, b, c, alpha, beta, gamma] = lna
        ralpha = alpha * math.pi/180
        rbeta = beta * math.pi/180
        rgamma = gamma * math.pi/180
        b1 = [a, 0., 0.]
        b2 = [b*math.cos(rgamma), b * math.sin(rgamma), 0.]
        b31 = c*math.cos(rbeta) + c * math.cos(ralpha)*math.cos(rgamma)
        b32 = c*math.cos(ralpha)*math.sin(rgamma)
        b33 = math.sqrt(c**2 - b31**2 - b32**2)
        b3 = [b31, b32, b33]
        self.lattice = [b1, b2, b3]

    def write_tinker_xyz(self, filename):
        f = open(filename, "w")
        f.write(str(self.natoms)+"  Tinker XYZ format\n")
        for i in range(6):
            f.write("   %.6f" % self.lna[i])
        f.write("\n")

        for i in range(0, self.natoms-1):
            f.write(" %3d  Re  %3.6f   %3.6f   %3.6f    75\n" %
                    (i+1, self.cartesianc[i][0], self.cartesianc[i][1],
                     self.cartesianc[i][2]))
        f.write(" %3d  Re  %3.6f   %3.6f   %3.6f    75" %
                (self.natoms, self.cartesianc[-1][0],
                 self.cartesianc[-1][1], self.cartesianc[-1][2]))
        f.close()

    def write_xyz(self, filename,):
        f = open(filename, "w")
        f.write(str(self.natoms)+'\n')
        f.write('Lattice="')
        for i in range(3):
            for j in range(3):
                f.write("%.6f  " % self.lattice[i][j])
        f.write('"')
        f.write(' Properties=species:S:1:pos:R:3 Time=0.0'+'\n')
        for i in range(0, self.natoms-1):
            f.write(" Re  %3.6f   %3.6f   %3.6f\n" % (self.cartesianc[i][0],
                                                      self.cartesianc[i][1],
                                                      self.cartesianc[i][2]))
        f.write(" Re  %3.6f   %3.6f   %3.6f" % (self.cartesianc[-1][0],
                                                self.cartesianc[-1][1],
                                                self.cartesianc[-1][2]))
        f.close()    

    def write_xsf(self, filename):
        f = open(filename, "w")
        f.write("CRYSTAL\n")
        f.write("PRIMVEC\n")
        for i in range(3):
            f.write("     %.6f    %.6f    %.6f\n" % (self.lattice[i][0],
                                                     self.lattice[i][1],
                                                     self.lattice[i][2]))
        f.write("PRIMCOORD\n")
        f.write(str(self.natoms)+"  1\n")

        for i in range(0, self.natoms-1):
            f.write("Re  %3.6f   %3.6f   %3.6f\n" % (self.cartesianc[i][0],
                                                     self.cartesianc[i][1],
                                                     self.cartesianc[i][2]))
        f.write("Re  %3.6f   %3.6f   %3.6f" % (self.cartesianc[-1][0],
                                               self.cartesianc[-1][1],
                                               self.cartesianc[-1][2]))
        f.close()

    def write_poscar(self, ctype, filename):
        # ctype = 'C' (Cartesian) or 'D' (Direct)
        f = open(filename, "w")
        f.write("POSCAR\n")
        f.write("1.000000\n")
        for i in range(3):
            f.write("%.6f   %.6f   %.6f\n" % (self.lattice[i][0],
                                              self.lattice[i][1],
                                              self.lattice[i][2]))
        f.write(str(self.natoms)+"\n")
        if ctype == 'C':
            f.write("Cartesian\n")
            for i in range(self.natoms-1):
                f.write("%.6f   %.6f   %.6f\n" % (self.cartesianc[i][0],
                                                  self.cartesianc[i][1],
                                                  self.cartesianc[i][2]))
            f.write("%.6f   %.6f   %.6f" % (self.cartesianc[-1][0],
                                            self.cartesianc[-1][1],
                                            self.cartesianc[-1][2]))
        if ctype == 'D':
            f.write("Direct\n")
            for i in range(self.natoms-1):
                f.write("%.6f   %.6f   %.6f\n" % (self.directc[i][0],
                                                  self.directc[i][1],
                                                  self.directc[i][2]))
            f.write("%.6f   %.6f   %.6f" % (self.directc[-1][0],
                                            self.directc[-1][1],
                                            self.directc[-1][2]))
        f.close()

    def write_lammps_data(self, filename, material):
        f = open(filename, "w")
        f.write("LAMMPS data file\n")
        f.write("\n")
        f.write(str(self.natoms)+" atoms\n")
        f.write("1 atom types\n")
        if abs(self.lattice[0][0]) > 0.1:
            f.write("0. %.5f xlo xhi\n" % (self.lattice[0][0]))
            f.write("0. %.5f ylo yhi\n" % (self.lattice[1][1]))
        else:
            f.write("0. %.5f xlo xhi\n" % (self.lattice[1][0]))
            f.write("0. %.5f ylo yhi\n" % (self.lattice[0][1]))
        f.write("0. %.5f zlo zhi\n" % (self.lattice[2][2]))
        f.write("%.5f %.5f %.5f xy xz yz\n\n" % (self.lattice[0][1],
                                                 self.lattice[0][2],
                                                 self.lattice[1][2]))

        if material == "Re":
                f.write("Masses\n\n")
                f.write("1 186.207\n\n")

        f.write("Atoms # atomic\n\n")
        for i in range(self.natoms - 1):
            f.write(str(i+1)+" 1 "+"%.5f   %.5f   %.5f\n" %
                    (self.cartesianc[i][0],
                     self.cartesianc[i][1],
                     self.cartesianc[i][2]))
        f.write(str(self.natoms) + " 1 "+"%.5f   %.5f   %.5f\n" %
                (self.cartesianc[-1][0],
                 self.cartesianc[-1][1],
                 self.cartesianc[-1][2]))
        f.close()

    def addstrain(self, def_tensor):  # F is the deformation gradient tensor
        self.lattice = np.dot(self.lattice, def_tensor)
        self.lna = self.get_lna()
        self.update_cartsc()
