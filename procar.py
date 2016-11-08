#!/usr/bin/env python

""" A python class for VASP PROCAR generated with LORBIT = 11. """

import numpy as np
import re

def findfloats(line):
    """ Find floats in a string

    only finds floats like the followings:
    1.0 1000 -1.0 +1 +1.0
    """
    # print line
    # floatpat =  r"\b[-+]?\d+\.\d+\b|\b[-+]?\d+\b"
    floatpat =  r"[-+]?\d+.\d+|[-+]?\d+"
    
    return re.findall(floatpat, line)

def Gaussian(x, x0, sigma=0.1):
    """
    Gaussian smearing
    """
    assert type(x) is np.ndarray
    return np.exp(-(x-x0)**2/(2*sigma**2))/np.sqrt(2 * np.pi * sigma**2) 

def Lorentzian(x, x0, sigma=0.1):
    """
    Lorentzian smearing
    """
    assert type(x) is np.ndarray
    return sigma / ((x-x0)**2 + sigma**2) / np.pi

class procar:
    """ A python class for VASP PROCAR generated with LORBIT = 11. """

    def __init__(self, inFile='PROCAR',
                 Efermi=0,
                 sigma=0.1, ismear=0,
                 extra=0.1, Ndot=500):
        """
        """

        self.FileName = inFile
        try:
            open(self.FileName) 
        except:
            raise

        # whether PROCAR has been read
        self.read = False
        # 
        self.ispin = None
        self.nkpts = None
        self.nbnds = None
        self.nions = None
        # the above variables are self-explanatory

        # lmax determines No. of columns in PROCAR partial weight
        self.ncols = None
        # energy levels & occ read from PROCAR
        self.eigs = None
        # k-points vector & weight read from PROCAR
        self.kpts = None
        # partial weight of each band 
        self.pwht = None

        self.emin = None
        self.emax = None
        self.extra = 0.1
        self.Ndot = Ndot
        self.Efermi = 0.0
        self.x =  None
        # total DOS
        self.tot = None
        # partial DOS
        self.par = None

        self.sigma = sigma
        self.ismear = ismear
        self.smear = (Gaussian, Lorentzian)[self.ismear]

    def readProcar(self):
        """ Reads VASP PROCAR file 

        Outputs:

            eigs:   Eigenvalues of bands
            kpts:   k-points vector and weight
            pwht:   partial weight of each band
            ndim:   (ispin, nkpts, nbands, nions)

        all of the above are numpy arrays
        """
        procar = [line for line in open(self.FileName).readlines()
                       if line.strip()          # exclude empty lines
                       if not 'dxy' in line]

        self.nkpts, self.nbnds, self.nions = [int(wd) for wd in procar[1].split() if wd.isdigit()]

        procar = procar[2:]

        eigs = [findfloats(line) for line in 
                filter(lambda x: 'band ' in x, procar)]
        kpts = [findfloats(line) for line in 
                filter(lambda x: 'k-point ' in x, procar)]
        # pwht = [line.split()[1:] for line in 
        #        filter(lambda x: (not 'energy' in x) and (not 'k-point' in x), procar)]
        pwht = np.asarray([line.split()[1:] for line in procar                                                                                                                  
                          if not re.search('[a-zA-Z]', line)], dtype=float)

        self.ncols = len(pwht[0])
        self.ispin = len(kpts) / self.nkpts

        ispin = self.ispin; nkpts = self.nkpts
        nbands= self.nbnds; nions = self.nions
        ncols = self.ncols

        # energy & occ in eigs
        self.eigs=np.array(eigs,dtype=float)[:,1:].ravel().reshape(ispin,nkpts,nbands,2)
        # kpt vector & weight in kpts
        self.kpts=np.array(kpts,dtype=float)[:,1:].ravel().reshape(ispin,nkpts,4)
        # partial weights
        self.pwht=np.array(pwht,dtype=float).ravel().reshape(ispin,nkpts,nbands,nions,ncols)

        self.read = True

    def genDOS(self, ions=None, sp=None, kp=None):
        """
        Generate DOS from procar.
        
        Output:
        tot: total DOS for each spin and each k-point
        par: partial DOS of selected ions for each spin and each k-point
        """

        if not self.read:
            self.readProcar()

        if ions is None:
            ions = np.arange(self.nions)
        else:
            ions = np.array(ions, dtype=int) - 1

        self.tot = np.zeros((self.ispin, self.nkpts, self.Ndot))
        self.par = np.zeros((self.ispin, self.nkpts, self.Ndot, self.ncols))

        emin = self.eigs[...,0].min()
        emax = self.eigs[...,0].max()
        erange = emax - emin

        if self.emin is None:
            self.emin = emin - erange * self.extra
        if self.emax is None:
            self.emax = emax + erange * self.extra
        self.x = np.linspace(self.emin, self.emax, self.Ndot)

        for i in range(self.ispin):
            for j in range(self.nkpts):
                if kp is None:
                    kwht = self.kpts[i,j,-1]
                else:
                    kwht = 1.0
                for k in range(self.nbnds):
                    tmp = kwht * self.smear(self.x, self.eigs[i,j,k,0], self.sigma)
                    # total dos for spin: i and k-points: j
                    self.tot[i, j, :] += tmp
                    # partial weight
                    w = np.sum(self.pwht[i,j,k,ions,:],axis=0)
                    # partial DOS
                    for ii in range(self.ncols):
                        self.par[i,j,:,ii] += tmp * w[ii]

        self.x -= self.Efermi
        if self.ispin == 2:
            self.tot[1,...] = -self.tot[1,...]
            self.par[1,...] = -self.par[1,...]

    def get_partial_weight(self, ions=None):
        """ get partial weight from PROCAR """

        if not self.read:
            self.readProcar()

        if ions is None:
            ions = np.arange(self.nions)
        else:
            ions = np.array(ions, dtype=int) - 1

        parw = np.zeros((self.ispin, self.nkpts, self.nbnds, 2))

        for i in range(self.ispin):
            for j in range(self.nkpts):
                for k in range(self.nbnds):
                    parw[i,j,k,0] = self.eigs[i,j,k,0]
                    parw[i,j,k,1] = np.sum(self.pwht[i,j,k,ions,-1])

        return parw

    def set_smear(self, ismear=0):
        """ set smearing method """
        self.smear = (Gaussian, Lorentzian)[ismear]

    def set_ndot(self, ndot=500):
        """ set number of points for smooth DOS plot """
        self.Ndot = ndot

    def set_emin(self, emin=-1.0):
        """ set lower bound for smooth DOS plot """
        self.emin = emin
        assert self.emax > self.emin

    def set_emax(self, emax=1.0):
        """ set upper bound for smooth DOS plot """
        self.emax = emax
        assert self.emax > self.emin

    def set_fermi(self, fermi=0.0):
        """ set Fermi level for the system """
        if self.x is not None:
            self.x = self.x + self.Efermi - fermi
        self.Efermi = fermi




    # def dosij(self, ions=None, s=None, k=None):
    #     """ spin decomposed DOS """

    #     # assert self.x is not None
    #     # assert self.tot is not None
    #     # assert self.par is not None

    #     for i in range(self.ispin):
    #         for j in range(self.nkpts):
    #             if k is None:
    #                 kwht = self.kpts[i,j,-1]
    #             else:
    #                 k = 1.0
    #             self.tot[i,j,:] *= kwht
    #            self.par[i,j,:,:] *= kwht
    #    if s is None:

