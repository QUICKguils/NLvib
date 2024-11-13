import numpy as np
import matplotlib.pyplot as plt
from scipy import io

# plt.rcParams['text.usetex'] = True
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['STIX Two Text'] + plt.rcParams['font.serif']
plt.rcParams['figure.figsize'] = (6.34,3.34)
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 200

class ASM:
    def __init__(self):

        self.nb = 3
        self.data = io.loadmat('res/group4_test1_'+ str(self.nb) + '.mat')

        self.dofs = [0,1]

        self.t = self.data['t'][0]
        self.x = self.data['x']
        self.v = self.data['xd']
        self.a = self.data['xdd']

        self.getRelativeFormat()

        self.tmin = [100,100]
        self.tmax = [250,250]

    def getRelativeFormat(self):
        """Express the relative position and velocity of the two dof."""

        self.x_rel = []
        self.v_rel = []

        self.x_rel.append(self.x[0] - self.x[1])
        self.x_rel.append(self.x[1] - self.x[0])
        self.v_rel.append(self.v[0] - self.v[1])
        self.v_rel.append(self.v[1] - self.v[0])

    def getDomainOfInterest(self):
        """Get the data in the domain of interest."""

        self.x_masked = []
        self.v_masked = []
        self.a_masked = []

        for (i,dof) in enumerate(self.dofs):
            self.mask = (self.t > self.tmin[i]) & (self.t < self.tmax[i])

            self.x_masked.append(self.x_rel[dof][self.mask])
            self.v_masked.append(self.v_rel[dof][self.mask])
            self.a_masked.append(self.a[dof][self.mask])

    def getZeroCrossing(self,vector,tol):
        """Get the index of the zero crossing of a vector."""
        idx = np.argwhere(np.abs(vector) < tol)
        return idx

    def excitation(self):
        pass

    def getSurface(self):
        """Get both curve of interest."""

        self.getDomainOfInterest()

        for dof in self.dofs:
            idx = self.getZeroCrossing(self.x_masked[dof],tol = 1e-5)
            v = self.v_masked[dof][idx]

            fig, ax = plt.subplots(1, 2)

            ax[0].scatter(v,-self.a_masked[dof][idx],marker='o',s=1)
            ax[0].set_xlabel(r'$\dot{x}$ [m/s]')
            ax[0].set_ylabel(r'- $\ddot{x}$ [m/s$^2$]')

            idx = self.getZeroCrossing(self.v_masked[dof], tol = 1e-2)
            x = self.x_masked[dof][idx]

            ax[1].scatter(x,-self.a_masked[dof][idx],marker='o',s=1)
            ax[1].set_xlabel(r' $x$ [m]')
            plt.tight_layout()
            plt.show()


def main():
    test = ASM()
    test.getSurface()


if __name__ == '__main__':
    main()
