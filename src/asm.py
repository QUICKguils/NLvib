"""Identification of nonlinearities with the acceleration surface method (ASM)."""

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
    def __init__(self,element):
        self.element = element

        self.nb = 1
        self.data = io.loadmat('DATA3/group4_test3_'+ str(self.nb) + '.mat')

        self.dofs = [0,1] #142-156

        self.t = self.data['t'][0]
        self.x = self.data['x']
        self.v = self.data['xd']
        self.a = self.data['xdd']

        self.Amp = [1,10,100]
        self.A = 50 #self.Amp[self.nb-1]

        if self.element == 1 or self.element == 3:
            self.getAbsoluteFormat()
        else:
            self.getRelativeFormat()

        self.tmin = [110,110]
        self.tmax = [140,140]

    def getAbsoluteFormat(self):
        """Express the absolute position and velocity of the two dof."""

        self.x_rel = []
        self.v_rel = []

        self.x_rel.append(self.x[0])
        self.x_rel.append(self.x[1])
        self.v_rel.append(self.v[0])
        self.v_rel.append(self.v[1])

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

    def getSurface(self):
        """Get both curve of interest."""

        self.getDomainOfInterest()

        if self.element == 1:
            self.dofs = [0]
        elif self.element == 3:
            self.dofs = [1]
        else:
            self.dofs = [0,1]

        for dof in self.dofs:
            idx = self.getZeroCrossing(self.x_masked[dof],tol = 1e-5)
            v = self.v_masked[dof][idx]

            fig, ax = plt.subplots(1, 2)

            if self.element == 1 and dof == 0:
                ax[0].scatter(v,-self.a_masked[dof][idx],marker='o',s=1)
                ax[0].set_xlabel(r'$\dot{q}_1$ [m / s]')
                ax[0].set_ylabel(r'- $\ddot{q}_1$ [m / s$^2$]')

            elif self.element == 3 and dof == 1:
                ax[0].scatter(v,-self.a_masked[dof][idx],marker='o',s=1)
                ax[0].set_xlabel(r'$\dot{q}_2$ [m / s]')
                ax[0].set_ylabel(r'- $\ddot{q}_2$ [m / s$^2$]')

            elif self.element == 2 and dof ==0 :
                ax[0].scatter(v,-self.a_masked[dof][idx],marker='o',s=1)
                ax[0].set_xlabel(r'$\dot{q}_1 - \dot{q}_2$ [m / s]')
                ax[0].set_ylabel(r'- $\ddot{q}_1$ [m / s$^2$]')

            elif self.element == 2 and dof ==1 :
                ax[0].scatter(v,-self.a_masked[dof][idx],marker='o',s=1)
                ax[0].set_xlabel(r'$\dot{q}_2 - \dot{q}_1$ [m / s]')
                ax[0].set_ylabel(r'- $\ddot{q}_2$ [m / s$^2$]')

            else :
                break

            ax[0].grid(True, linewidth=0.5, alpha = 0.3)
            ax[0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))

            idx = self.getZeroCrossing(self.v_masked[dof], tol = 1e-3)
            x = self.x_masked[dof][idx]

            if self.element == 1 and dof == 0:
                ax[1].scatter(x,-self.a_masked[dof][idx],marker='o',s=1)
                ax[1].set_xlabel(r'$q_1$ [m]')

            elif self.element == 3 and dof == 1:
                ax[1].scatter(x,-self.a_masked[dof][idx],marker='o',s=1)
                ax[1].set_xlabel(r'$q_2$ [m]')

            elif self.element == 2 and dof ==0 :
                ax[1].scatter(x,-self.a_masked[dof][idx],marker='o',s=1)
                ax[1].set_xlabel(r'$q_1 - q_2$ [m]')

            elif self.element == 2 and dof ==1 :
                ax[1].scatter(x,-self.a_masked[dof][idx],marker='o',s=1)
                ax[1].set_xlabel(r'$q_2 - q_1$ [m]')

            else :
                break

            ax[1].grid(True, linewidth=0.5, alpha = 0.3)
            ax[1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            plt.tight_layout()
            #plt.savefig('ASM_element_' + str(self.element) + '_dof_' + str(dof) + '.pdf', bbox_inches='tight',format = 'pdf')
            #plt.savefig('ASM_element_' + str(self.element) + '_dof_' + str(dof) + '.png', bbox_inches='tight',format = 'png',dpi = 2000)
            plt.show()


def main():
    test = ASM(element = 1)
    test.getSurface()


if __name__ == '__main__':
    main()
