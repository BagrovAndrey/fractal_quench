import numpy as np
from scipy.sparse import diags, kron
from scipy.sparse.linalg import eigsh, eigs
from numpy.linalg import eig
import matplotlib.pyplot as plt
import time
from numpy import linalg as LA
from mpl_toolkits import mplot3d
import sys

def cartesian(triple_code):

    code_length = len(triple_code)
    scales = []

    for iloop in range(1, code_length):
        scales.append(0.5**iloop)

    scales.append(0.5**(code_length-1))

    x = 0
    y = 0

    for iloop in range(code_length):

        if triple_code[iloop] == 0:
            x += 0
            y += scales[iloop]*1

        if triple_code[iloop] == 1:
            x += scales[iloop]*(-np.sqrt(3.)/2)
            y += scales[iloop]*(-1./2)

        if triple_code[iloop] == 2:
            x += scales[iloop]*(np.sqrt(3.)/2)
            y += scales[iloop]*(-1./2)

    return x, y

def side(l):

    for iloop in range(l):

        if iloop == 0:
            x = 2
        else:
            x = 2*x-1

    return x

def Sierp_gasket(N):

    n_tot=int((3**N+3)/2)#number_of_sites
    array=[(0,1,2)]#initial array of triangles
    tuple_coord = [[]]
    vertex_fractal = [None]*n_tot

    j=3
    "#create array of triangles on a new iteration"
    for k in range(N-1):
        array_next = []
        tuple_coord_next = []

        for el in array:
            ind = array.index(el)

            array_next.append((el[0], j, j+2))
            tuple_coord_next.append(tuple_coord[ind]+[0])

            array_next.append((j, el[1], j + 1))
            tuple_coord_next.append(tuple_coord[ind]+[1])

            array_next.append((j+2, j+1, el[2]))
            tuple_coord_next.append(tuple_coord[ind]+[2])

            j+=3

        array=array_next.copy()
        tuple_coord=tuple_coord_next.copy()

    for triangle in array:

        ind = array.index(triangle)
        for el in triangle:

            if vertex_fractal[el] == None:
                vertex_fractal[el] = tuple_coord[ind]+[triangle.index(el)]

    Sierp_matrix=np.zeros((n_tot, n_tot))
    #print(n_tot, array)
    "create adjacency matrix of array of triangles"
    for tuple in array:
        Sierp_matrix[tuple[0],tuple[1]] = 1
        Sierp_matrix[tuple[1], tuple[2]] = 1
        Sierp_matrix[tuple[2], tuple[0]] = 1

        Sierp_matrix[tuple[1],tuple[0]] = 1
        Sierp_matrix[tuple[2], tuple[1]] = 1
        Sierp_matrix[tuple[0], tuple[2]] = 1

    return Sierp_matrix, vertex_fractal

def evolved_spectrum(vec, t):

    result = map(lambda x: np.exp(1j*t*x), vec)

    return list(result)
 
def main():

    N=5
    n_frac=int((3**N+3)/2)
    
    gasket = Sierp_gasket(N)

    Ad_fractal=gasket[0]
    cartesian_codes=gasket[1]

    x_fractal = []
    y_fractal = []    

    for point in cartesian_codes:
        x_fractal.append(cartesian(point)[0])
        y_fractal.append(cartesian(point)[1])

    #print("x_fractal = ", x_fractal)
    #print("y_fractal = ", y_fractal)

    # Create complementary set of points

    x_vec = np.array([(max(x_fractal) - min(x_fractal))/(side(N)-1), 0])
    y_vec = np.array([(max(x_fractal) - min(x_fractal))/(side(N)-1)*0.5, (max(x_fractal) - min(x_fractal))/(side(N)-1)*np.sqrt(3)/2])

    unit_vec = np.sqrt(x_vec[0]**2 + x_vec[1]**2)
 
    print("unit_vec = ", unit_vec)

    #print("Xvec", x_vec)
    #print("Yvec", y_vec)

    x_full = []
    y_full = []

    for iloop in range(side(N)):
        for jloop in range(side(N)-iloop):

            #complement.append(iloop*x_vec + jloop*y_vec)
            x_full.append(min(x_fractal) + (iloop*x_vec + jloop*y_vec)[0])
            y_full.append(min(y_fractal) + (iloop*x_vec + jloop*y_vec)[1])

    # Introduce points complementing the fractal to integer dimension lattice

    rounding = 7

    fractal_coords = [[round(x_fractal[iloop], rounding), round(y_fractal[iloop], rounding)] for iloop in range(len(x_fractal))]
    full_coords = [[round(x_full[iloop], rounding), round(y_full[iloop], rounding)] for iloop in range(len(x_full))]

    # Sorting coordinates so that the complement appears after the fractal part

    complement_coords = [x for x in full_coords if x not in fractal_coords]
    full_coords = fractal_coords + complement_coords

    x_complement = [x[0] for x in complement_coords]
    y_complement = [y[1] for y in complement_coords]

    x_full = [x[0] for x in full_coords]
    y_full = [y[1] for y in full_coords]

    n_frac = len(fractal_coords)
    n_full = len(full_coords)

    # Plot connections in the fractal

    for iloop in range(n_frac):
        for jloop in range(n_frac):

            if Ad_fractal[iloop][jloop] == 1.0:

                plt.plot([x_fractal[iloop], x_fractal[jloop]], [y_fractal[iloop], y_fractal[jloop]], color='green')

    #plt.show()

    # Adjacency matrix of the complete graph

    J_frac = 1.0
    J_aux = 0.9999
    J_comp = 0.9998
    J_interm = 0.9997

    Ad_complete = np.zeros((n_full, n_full))
    Ad_complete[:Ad_fractal.shape[0],:Ad_fractal.shape[1]] = Ad_fractal # The inner block of the adjacency matrix represents the original fractal adj. matrix

    # Fractal adjacency matrix that will be used in the quench protocol

    Ad_fractal_ext = Ad_complete 
    
    for iloop in range(n_frac, n_full):

        Ad_fractal_ext[iloop][iloop] = 1.

    print("Ad_fractal = ", Ad_fractal[15])
    print("Ad_complete = ", Ad_complete[15])

    # Connect complementing points

    for iloop in range(n_frac, n_full):
        for jloop in range(n_frac, n_full):

            if abs((full_coords[iloop][0] - full_coords[jloop][0])**2 + (full_coords[iloop][1] - full_coords[jloop][1])**2 - unit_vec**2) < 0.1*unit_vec**2:

                Ad_complete[iloop][jloop] = J_comp

    for iloop in range(n_frac, n_full):
        for jloop in range(n_frac, n_full):

            if Ad_complete[iloop][jloop] == J_comp:

                plt.plot([full_coords[iloop][0], full_coords[jloop][0]], [full_coords[iloop][1], full_coords[jloop][1]], color='red')

    # plt.show()


    # Connect complementing and fractal points

    for iloop in range(n_frac):
        for jloop in range(n_frac, n_full):

            if abs((full_coords[iloop][0] - full_coords[jloop][0])**2 + (full_coords[iloop][1] - full_coords[jloop][1])**2 - unit_vec**2) < 0.1*unit_vec**2:

                Ad_complete[iloop][jloop] = J_interm

    for iloop in range(n_frac):
        for jloop in range(n_frac, n_full):

            if Ad_complete[iloop][jloop] == J_interm:

                plt.plot([full_coords[iloop][0], full_coords[jloop][0]], [full_coords[iloop][1], full_coords[jloop][1]], color='blue')

    # plt.show()

    # Connect fractal points that have not been connected before

    for iloop in range(n_frac):
        for jloop in range(n_frac):

            if Ad_complete[iloop][jloop] == 0.0 and abs((full_coords[iloop][0] - full_coords[jloop][0])**2 + (full_coords[iloop][1] - full_coords[jloop][1])**2 - unit_vec**2) < 0.1*unit_vec**2:

                Ad_complete[iloop][jloop] = J_aux

    for iloop in range(n_frac):
        for jloop in range(n_frac):

            if Ad_complete[iloop][jloop] == J_aux:

                plt.plot([full_coords[iloop][0], full_coords[jloop][0]], [full_coords[iloop][1], full_coords[jloop][1]], color='black')

    plt.show()
    print("Ad_complete = ", Ad_complete)

    # Obtain ground state on the homogeneous lattice

    spec_comp, Q_comp = LA.eig(Ad_complete)
    Q_comp_inv = LA.inv(Q_comp)

    psi0 = np.ones(n_full)/np.sqrt(n_full)
    #psi0[0] = 1.

    # For fractal time evolution

    spec_frac, Q_frac = LA.eig(Ad_fractal_ext)
    Q_frac_inv = LA.inv(Q_frac)

    #spec, Q = LA.eig(Ad_fractal)
    #Qinv = LA.inv(Q)

    # sys.exit()

    for dt in range(0, 50):

        print(dt)

    #    time_spectrum = evolved_spectrum(spec_comp, dt/100)
    #    psi_new=np.round(np.einsum('ij,jk,kl,l->i', Q_comp, np.diag(time_spectrum), Q_comp_inv, psi0),3)

        time_spectrum = evolved_spectrum(spec_frac, dt/100)
        psi_new=np.einsum('ij,jk,kl,l->i', Q_frac, np.diag(time_spectrum), Q_frac_inv, psi0)

        psi_density = np.array(list(map(lambda x: np.abs(x)**2, psi_new)))

        print(sum(psi_density))

        plt.scatter(x_full, y_full, c=psi_density, cmap='Oranges')
        plt.savefig(str(dt)+'.png')


#    for dt in range(1, 500):

#        time_spectrum = evolved_spectrum(spec, dt/10)
#        psi_new=np.einsum('ij,jk,kl,l->i', Q, np.diag(time_spectrum), Qinv, psi0)
#        psi_density = list(map(lambda x: np.abs(x)**2, psi_new))

#        print(dt)

#        plt.scatter(x_fractal, y_fractal, c=psi_density, cmap='Oranges')
#        plt.savefig(str(dt)+'.png')

    return

main()
