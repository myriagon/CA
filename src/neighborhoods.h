/*
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
*/


// Neighborhood types

// von Neumann is a subset of Moore, which is a subset of extended Moore

//  e e e e e
//  e m v m e
//  e v v v e
//  e m v m e
//  e e e e e

// Keep neighborhood-type numbers in the range [0, 255], at least for now.
// The numbers must be non-negative since they are used as indices into the
// neighborhood_type_names array.
// Neighborhood type is treated as an unsigned 8-bit int by hodge-podge.c
// and is stored as an unsigned 16-bit int attribute in the output HDF5 file.

enum neighborhood_type_nums { VON_NEUMANN=0, MOORE, EXT_MOORE };

const char *neighborhood_type_names [3] = {
   "von Neumann", "Moore", "extended Moore"
};

const uint8 num_neighbors [3] = { 5, 9, 25 };

// row & column offsets from cell

const sint8 von_neumann_y [5] = {                        -1,            0, 0, 0,            1 };
const sint8 von_neumann_x [5] = {                         0,           -1, 0, 1,            0 };

const sint8 moore_y       [9] = {                     -1,-1,-1,         0, 0, 0,         1, 1, 1 };
const sint8 moore_x       [9] = {                     -1, 0, 1,        -1, 0, 1,        -1, 0, 1 };

const sint8 ext_moore_y  [25] = { -2,-2,-2,-2,-2,  -1,-1,-1,-1,-1,   0, 0, 0, 0, 0,   1, 1, 1, 1, 1,   2, 2, 2, 2, 2 };
const sint8 ext_moore_x  [25] = { -2,-1, 0, 1, 2,  -2,-1, 0, 1, 2,  -2,-1, 0, 1, 2,  -2,-1, 0, 1, 2,  -2,-1, 0, 1, 2 };


