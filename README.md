# QuICTools
A collection of various functions that are used frequently in numerical simulations in the Quantum Information and Control Group at the University of Arizona.

The quictools package is a collection of convenience functions that are used frequently in numerical simulations. The quictools package contains 5 subpackages.

`quictools.super_operators` contains functions that deal with converting matrices to flattened vectorized representations and converting linear operators to equivalent superoperator representations. There are also functions to generate a basis of generalized Gell-Mann matrices which are a basis of orthogonal Hermitian basis elements.

`quictools.stats` contains functions to generate random states, random Hermitian matrices, and random unitary matrices as a function of Hilbert space dimension.

`quictools.spins` contains functions to generate matrix representations of spin operators, a function to efficiently calculate spin coherent states in high dimensions, and various functions to generate random Hermitian matrice, unitary matrices, or states in terms of a desired spin size. The random functions are wrappers that reimplement the functions of `quictools.stats` so that the argument is the spin size of a system instead of the Hilbert space dimension.

`quictools.quantum` contains functions to implement simple numerical simulations of quantum states in the presence of errors.

`quictools.models` contains functions to calculate Hamiltonians, unitary matrices, and some classical simulations for specific models that are used frequently in the Quantum Information and Control Group.
