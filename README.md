# CA

## Overview

This project implements the "hodge podge machine" cellular automaton described
by Gerhardt and Schuster [1] using C, MPI, and HDF5.

This is experimental work in progress.

I created the project with three goals in mind:

- I wanted to implement the hodge podge machine and explore its behavior.

- I wanted to practice parallel programming with C and the Message Passing
Interface (MPI), particularly the tricky business of halo exchange.

- I wanted to experiment with running parallel software in the Amazon Web
Services (AWS) cloud.

The software works and I have used it to produce captivating movies of the
hodge podge machine evolving.  There is still plenty to do when I find time,
including documenting how to build and run the software and posting a gallery
of movies.  See TODO.txt.

## Details

The lattice of cells is divided into a matrix of tiles.  Each MPI rank
(process) works on a single tile and writes data and metadata to a separate
HDF5 file.  Since the evolution of a cell depends on the states of neighboring
cells, MPI ranks have to communicate about cells on their mutual borders.
Therefore they have to proceed in lockstep and share data via halo exchange.

## References

[1]  A cellular automaton describing the formation of spatially ordered
structures in chemical systems.  Gerhardt M, Schuster H.
Physica D 36 (1989) 209-221.

## License

Copyright (c) 2017-2020 Scott Davis

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Please acknowledge my work if it is useful to you.


