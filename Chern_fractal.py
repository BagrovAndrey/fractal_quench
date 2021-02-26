import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from numpy import linalg as LA

def three_regions(N, pos, L):

    n_tot = int((3 ** N + 3) / 2)
    l1=1.2*L/4
    l2=L/2+l1
    A=[]
    B=[]
    C=[]
    #"division by region as in 1807.00367"
    for i in range(n_tot):
        if pos[i,0]>(L/2-0.0000001) and pos[i,1]>=l1/3 and pos[i,1]<-pos[i,0]+l2 \
                and pos[i,1]>np.sqrt(3)*pos[i,0]+(l1/3-np.sqrt(3)*(L/2+l1/3)):
            A.append(i)
        if pos[i,0]>(L/2-0.0000001) and pos[i,1]<l1/3 and pos[i,1]>(0-0.0000001)\
                and pos[i, 1] <= -np.sqrt(3) * pos[i, 0] + (l1 / 3 + np.sqrt(3) * (L / 2 + l1 / 3)):
            B.append(i)
        if pos[i,1]>(0-0.0000001) and pos[i,1]<-pos[i,0]+l2 \
                and pos[i, 1] <= np.sqrt(3) * pos[i, 0] + (l1 / 3 - np.sqrt(3) * (L / 2 + l1 / 3))\
                and pos[i, 1] > -np.sqrt(3) * pos[i, 0] + (l1 / 3 + np.sqrt(3) * (L / 2 + l1 / 3)):
            C.append(i)

    return A,B,C

"Chern number on given energy"
def Chern_number(E, N,B,L):

    Ad_matrix, pos =Sierp_gasket(N,B,L)
    spectra, vectors = eigh(Ad_matrix)

    n_tot = int((3 ** N + 3) / 2)

    "index of energy"
    n_e=0
    while spectra[n_e]<E:
        n_e+=1
    n_e=n_e-1

    "building of projector"
    P=np.zeros((n_tot, n_tot), dtype=complex)
    for i in range(n_e):
        psi = np.tile(vectors[:, i], (n_tot, 1))
        psi_conj = np.transpose(np.tile(vectors[:, i], (n_tot, 1)))
        P += np.conjugate(psi_conj) * psi

    A, B, C=three_regions(N, pos, L)
    "calculating Chern number"
    Chern=0
    for j in A:
        for k in B:
            for l in C:
                Chern+=np.real(12*np.pi*1j*(P[j,k]*P[k,l]*P[l,j]-P[j,l]*P[l,k]*P[k,j]))
    return Chern

"Chern numbers on the spectrum range"
def Chern_numbers(energies,N,B, L):

    Ad_matrix, pos =Sierp_gasket(N,B,L)
    spectra, vectors = eigh(Ad_matrix)

    n_tot = int((3 ** N + 3) / 2)
    A, B, C = three_regions(N, pos, L)
    Cherns=np.zeros(len(energies))

    "index of energy"
    n_e=0
    counter=0
    while spectra[n_e]<energies[0]:
        n_e+=1
        counter += 1
    if counter > 0:
        n_e = n_e - 1

    "building of projector"
    P=np.zeros((n_tot, n_tot), dtype=complex)
    for i in range(n_e):
        psi = np.tile(vectors[:, i], (n_tot, 1))
        psi_conj = np.transpose(np.tile(vectors[:, i], (n_tot, 1)))
        P += np.conjugate(psi_conj) * psi

    "calculating Chern number"
    for j in A:
        for k in B:
            for l in C:
                Cherns[0]+=np.real(12*np.pi*1j*(P[j,k]*P[k,l]*P[l,j]-P[j,l]*P[l,k]*P[k,j]))

    for i in range(len(energies)-1):
        print("i", i)
        n_e_p=n_e
        counter=0
        while spectra[n_e] < energies[i+1]:
            n_e += 1
            if n_e>(len(spectra)-1):
                n_e=len(spectra)-1
                break
            counter+=1

        if counter>0:
            n_e=n_e-1

        if n_e==n_e_p:
            Cherns[i+1]=Cherns[i]
            continue

        for j in range(n_e-n_e_p):
            psi = np.tile(vectors[:,n_e_p+j+1], (n_tot, 1))
            psi_conj = np.transpose(np.tile(vectors[:,n_e_p+j+1], (n_tot, 1)))
            P += np.conjugate(psi_conj) * psi

        "calculating Chern number"
        for j in A:
            for k in B:
                for l in C:
                    Cherns[i+1] += np.real(12 * 1j*np.pi * (P[j, k] * P[k, l] * P[l, j] - P[j, l] * P[l, k] * P[k, j]))

    plt.title("Chern numbers Sierpinski gasket")
    plt.plot(energies, Cherns)
    plt.savefig("Chern_numbers.png")

    return Cherns

def Sierp_gasket(N, B, L):

    n_tot=int((3**N+3)/2)#number_of_sites
    array=[(0,1,2)]#initial array of triangles
    pos=np.zeros((n_tot,2))
    pos[:3]=[[0,0],[0,L],[L,0]]
    j=3
    "#create array of triangles on a new iteration"
    for k in range(N-1):
        array_next = []
        for tuple in array:
            array_next.append((tuple[0], j, j+2))
            array_next.append((j, tuple[1], j + 1))
            array_next.append((j+2, j+1, tuple[2]))
            pos[j]=[pos[tuple[0],0],(pos[tuple[0],1]+pos[tuple[1],1])/2]
            pos[j+1]=[(pos[tuple[0],0]+pos[tuple[2],0])/2,(pos[tuple[0],1]+pos[tuple[1],1])/2]
            pos[j+2]=[(pos[tuple[0],0]+pos[tuple[2],0])/2, pos[tuple[0],1]]
            j+=3
        array=array_next.copy()

    Sierp_matrix=np.zeros((n_tot, n_tot),dtype=complex)

    "create adjacency matrix of array of triangles"
    for tuple in array:
        Sierp_matrix[tuple[0],tuple[1]]=-np.exp(1j*B*(pos[tuple[1],1]-pos[tuple[0],1])*(pos[tuple[1],0]+pos[tuple[0],0])/2)
        Sierp_matrix[tuple[1], tuple[2]] = -np.exp(1j*B*(pos[tuple[2],1]-pos[tuple[1],1])*(pos[tuple[2],0]+pos[tuple[1],0])/2)
        Sierp_matrix[tuple[2], tuple[0]] = -np.exp(1j*B*(pos[tuple[0],1]-pos[tuple[2],1])*(pos[tuple[0],0]+pos[tuple[2],0])/2)

        Sierp_matrix[tuple[1],tuple[0]]=-np.exp(-1j*B*(pos[tuple[1],1]-pos[tuple[0],1])*(pos[tuple[1],0]+pos[tuple[0],0])/2)
        Sierp_matrix[tuple[2], tuple[1]] = -np.exp(-1j*B*(pos[tuple[2],1]-pos[tuple[1],1])*(pos[tuple[2],0]+pos[tuple[1],0])/2)
        Sierp_matrix[tuple[0], tuple[2]] = -np.exp(-1j*B*(pos[tuple[0],1]-pos[tuple[2],1])*(pos[tuple[0],0]+pos[tuple[2],0])/2)

    return Sierp_matrix, pos

def show_butterfly(N, L):

    x=[]
    y=[]
    for i in range(101):
        print("step of butterfly", i)
        B=0+2*np.pi*(2**(2*N-2)/L**2)*i/100 #(2**(2*N-2)/L**2) is area of the smalles triangle and 2pi is full phase
        Ad_matrix, pos = Sierp_gasket(N, B, L)
        spectra=LA.eigvalsh(Ad_matrix)
        x.append(spectra)
        y.append(B*np.ones(len(spectra)))

    plt.title("Hofstadter butterfly Sierpinski gasket")
    plt.hexbin(x, y, gridsize=(200,70),bins='log')
    plt.savefig("butterfly.png")
    plt.close()

def show_spectrum(N, B, L):

    Ad_matrix, pos = Sierp_gasket(N, B, L)
    spectra = LA.eigvalsh(Ad_matrix)
    plt.figure()
    n, bins, patches = plt.hist(spectra, bins=100, facecolor='b')
    plt.title("spectrum Sierpinski gasket")
    plt.draw()
    plt.savefig("spectrum Sierpinski gasket.png")
    plt.close()

def domain_position(index,pos):
    dom_pos_x=[]
    dom_pos_y = []
    for i in index:
        dom_pos_x.append(pos[i,0])
        dom_pos_y.append(pos[i, 1])
    return dom_pos_x, dom_pos_y

def show_fractal(N,L):

    Ad_matrix, pos = Sierp_gasket(N, 0, L)
    A, B, C = three_regions(N, pos, L)
    pos_A_x,pos_A_y =domain_position(A,pos)
    pos_B_x,pos_B_y =domain_position(B,pos)
    pos_C_x,pos_C_y =domain_position(C,pos)

    plt.scatter(pos[:,0], pos[:,1], s=4,alpha=0.5, color="grey")
    plt.scatter(pos_A_x, pos_A_y, s=4, color="red", label="A")
    plt.scatter(pos_B_x, pos_B_y, s=4, color="green", label="B")
    plt.scatter(pos_C_x, pos_C_y, s=4, color="blue", label="C")

    plt.legend()
    plt.savefig("Sierpinski_gasket.png")
    plt.close()


def main():

    N=7
    L=2
    B=0.25*2*np.pi*(2**(2*N-2)/L**2)#magnetic flux=0.25 as in 1807.00367

    show_fractal(N, L)
    #show_butterfly(N,L)
    #show_spectrum(N, B, L)

    energies=np.linspace(-4,4,400)
    Chern_numbers(energies,N,B, L)

main()