# CA

## Overview

This project is about exploring *cellular automata*, experimenting with software
technologies, making pretty pictures and movies, possibly doing science, and
having fun!

Two specific desires spurred the author.
He wanted to implement the *"hodge podge machine"* described by Gerhardt and
Schuster [1] and to explore its behavior.
He also wanted to experiment with *MPI programming* in the context of *Amazon
Web Services CloudFormation*.

The code was developed and tested with Amazon Linux, Open MPI, HDF5, and the
SGE parallel execution environment.

The lattice of cells is divided into a "matrix" of "tiles".  Each MPI rank
(process) works on a single tile and writes data and metadata to a separate
HDF5 file.  Since the evolution of a cell depends on the states of neighboring
cells, MPI ranks have to communicate about cells on their mutual borders.
Therefore they have to proceed in lockstep and share data via halo exchange.

Reference:
[1]  A cellular automaton describing the formation of spatially ordered
structures in chemical systems.  Gerhardt M, Schuster H.
Physica D 36 (1989) 209-221

