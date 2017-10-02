/*
   Copyright 2017 S Davis
*/


// Neighborhood types

// von Neumann is a subset of Moore, which is a subset of extended Moore

//  e e e e e
//  e m v m e
//  e v v v e
//  e m v m e
//  e e e e e

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

