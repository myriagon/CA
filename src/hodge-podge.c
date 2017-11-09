/*
   Implement a cellular automaton using MPI and HDF5

   Specifically, implement "hodge podge machine" described in this paper:
   A cellular automaton describing the formation of spatially ordered
   structures in chemical systems.  Gerhardt M, Schuster H.
   Physica D 36 (1989) 209-221

   Copyright 2017 S Davis
*/


#include <stdio.h>
#include <stdlib.h>   // calloc (), exit ()
#include <string.h>   // memset (), strlen (), strncpy ()
#include <mpi.h>
#include <openssl/rand.h>   // RAND_bytes ()
#include "hdf5.h"

#include "my-types.h"
#include "neighborhoods.h"


#define USE_SZIP 0

#define ZLIB_COMPRESSION_LEVEL 6


// Choose little-endian byte order for the HDF5 file since we're most likely to
// run on little-endian x86-64 processors


// Dataspace has 3 dimensions: steps, tile_y_dim, tile_x_dim
#define N_DIMS 3


// MPI halo exchange

enum { ABOVE=0, BELOW, LEFT, RIGHT, UL, UR, LL, LR };

#define N_HALO_NEIGHBORS 8

static size_t directions [N_HALO_NEIGHBORS] = {
   ABOVE, BELOW, LEFT, RIGHT, UL, UR, LL, LR
};

static size_t reverse_directions [N_HALO_NEIGHBORS] = {
   BELOW, ABOVE, RIGHT, LEFT, LR, LL, UR, UL
};


// Function declarations

int halo_setup (int rank,
   uint16 tile_y_dim, uint16 tile_x_dim, uint16 matrix_y_dim, uint16 matrix_x_dim,
   uint16 *matrix_y, uint16 *matrix_y_above, uint16 *matrix_y_below,
   uint16 *matrix_x, uint16 *matrix_x_left,  uint16 *matrix_x_right,
   int *halo_ranks, int *halo_nelems, uint16 *rcv_mem[], uint16 *snd_mem[]);

int halo_exchange (int rank,
   uint16 tile_y_dim, uint16 tile_x_dim,
   uint16 *tile_mem,
   uint16 tile_plus_halo_y_dim, uint16 tile_plus_halo_x_dim,
   uint16 *tile_plus_halo_mem,
   int *halo_ranks, int *halo_nelems, uint16 *rcv_mem[], uint16 *snd_mem[]);

static int create_u16le_attr (int rank, hid_t dset_id, const char *att_name, uint16 att_val);
static int create_u32le_attr (int rank, hid_t dset_id, const char *att_name, uint32 att_val);

static int gen_random_uint16_0_to_V (int rank, uint16 V, size_t len, uint16 *mem);

static int evolve_hodge_podge (int rank, uint32 step, uint8 neighborhood_type,
   uint16 V, uint16 g, uint16 k1, uint16 k2,
   uint32 tile_plus_halo_y_dim, uint32 tile_plus_halo_x_dim,
   uint16 *tile_plus_halo_mem,
   uint32 tile_y_dim, uint32 tile_x_dim,
   uint16 *tile_mem);

static int write_step (int rank, hid_t dset_id, uint32 step, uint32 y_dim, uint32 x_dim, uint16 *step_mem);


/*------------------------------------------------------------------------------
main
------------------------------------------------------------------------------*/

int main (int argc, char **argv)
{
   // MPI stuff
   int rank = -1;

   // HDF5 stuff
   hid_t file_id;
   hsize_t dims[N_DIMS];
   hid_t dataspace_id;
   hid_t dataset_id;
   herr_t status;
   int err = 0;

   // Attributes
   uint16 matrix_x_dim = 0, matrix_y_dim = 0;
   uint32 tile_x_dim = 0, tile_y_dim = 0;
   uint32 steps = 0;
   uint8  neighborhood_type = 0;
   uint16 V = 0;
   uint16 g = 0;
   uint16 k1 = 0;
   uint16 k2 = 0;
   int fatal = 0;

#define MAX_PATH 1023  // ???
   char fpath[MAX_PATH+1];
   char fsufx[MAX_PATH+1];  // file suffix _%d.h5 with MPI rank


   // Clear file path
   memset (fpath, '\0', MAX_PATH+1);
   memset (fsufx, '\0', MAX_PATH+1);


   // Parse command-line.  I'm keeping this part simple for the moment because
   // I want to focus on other aspects of the program.  I'll come back to this
   if (10 != argc) {
      fprintf (stderr, "Usage:\n"
         "%s matrix-dims tile-dims time-steps neighborhood-type V g k1 k2"
         " output-file-path\n\n"
         "Example:\n"
         "%s 3x2 1280x720 10000 0 100 7 2 3 /data/V_100_g_7_k1_2_k2_3\n"
         "Note:\n"
         "_<mpi-rank>.h5 will be appended to each output file name\n\n",
            argv[0], argv[0]);
      exit (1);
   }
   fatal = 0;
   if ( 2 != sscanf (argv[1], "%hux%hu", &matrix_x_dim, &matrix_y_dim) ) {
      fprintf (stderr, "Failed to parse arg 1, matrix dimensions\n");
      fatal++;
   }
   if ( 2 != sscanf (argv[2], "%ux%u", &tile_x_dim, &tile_y_dim) ) {
      fprintf (stderr, "Failed to parse arg 2, tile dimensions\n");
      fatal++;
   }
   if ( 1 != sscanf (argv[3], "%u", &steps) ) {
      fprintf (stderr, "Failed to parse arg 3, time steps\n");
      fatal++;
   }
   if ( 1 != sscanf (argv[4], "%hhu", &neighborhood_type) ) {
      fprintf (stderr, "Failed to parse arg 4, neighborhood type (0, 1, or 2)\n");
      fatal++;
   }
   if ((uint8)EXT_MOORE != neighborhood_type) {
      fprintf (stderr, "Sorry, only neighborhood type %hhu (%s) is supported"
         " right now\n", (uint8)EXT_MOORE, neighborhood_type_names [EXT_MOORE]);
      fatal++;
   }
   if ( 1 != sscanf (argv[5], "%hu", &V) ) {
      fprintf (stderr, "Failed to parse arg 5, V\n");
      fatal++;
   }
   if ( 1 != sscanf (argv[6], "%hu", &g) ) {
      fprintf (stderr, "Failed to parse arg 6, g\n");
      fatal++;
   }
   if ( 1 != sscanf (argv[7], "%hu", &k1) ) {
      fprintf (stderr, "Failed to parse arg 7, k1\n");
      fatal++;
   }
   if ( 1 != sscanf (argv[8], "%hu", &k2) ) {
      fprintf (stderr, "Failed to parse arg 8, k2\n");
      fatal++;
   }
   if ( strlen (argv[9]) > MAX_PATH ) {
      fprintf (stderr, "Output file path is too long\n");
      fatal++;
   }
   strncpy (fpath, argv[9], MAX_PATH);
   if (fatal) {
      exit (3);
   }

   // !!! Perform sanity checks on parameters !!!


   // Set up MPI and get the rank of this process

   MPI_Init (&argc, &argv);

   MPI_Comm_rank (MPI_COMM_WORLD, &rank);


   // Have just one rank print parameters
   if (0 == rank) {
      fprintf (stderr, "Parameters:\n"
         "matrix dimensions %hu by %hu (x by y)\n"
         "tile dimensions %u by %u (x by y)\n"
         "time steps %u\n"
         "neighborhood type %hhu (%s)\n"
         "V %hu g %hu k1 %hu k2 %hu\n",
            matrix_x_dim, matrix_y_dim,
            tile_x_dim, tile_y_dim,
            steps,
            neighborhood_type, neighborhood_type_names [neighborhood_type],
            V, g, k1, k2);
   }


   // Compose file suffix, _%d.h5, with MPI rank.  Append it to file path.
   // Check length
   sprintf (fsufx, "_%d.h5", rank);
   if (strlen (fpath) + strlen (fsufx) > MAX_PATH) {
      fprintf (stderr, "output file path with _<mpi-rank>.h5 suffix is too long\n");
      exit (5);
   }
   strcat (fpath, fsufx);


   // Halo exchange setup

   uint16 matrix_y       = -1, matrix_x       = -1;
   uint16 matrix_y_above = -1, matrix_y_below = -1;
   uint16 matrix_x_left  = -1, matrix_x_right = -1;

   int halo_ranks  [N_HALO_NEIGHBORS];  // array of MPI ranks
   int halo_nelems [N_HALO_NEIGHBORS];  // array containing numbers of data elements

   // Memory for halo data
   uint16 *rcv_mem [N_HALO_NEIGHBORS];  // array of pointers
   uint16 *snd_mem [N_HALO_NEIGHBORS];  // array of pointers

   // Initialize halo arrays
   for (int j = 0; j < N_HALO_NEIGHBORS; j++) {
      halo_ranks  [j] = 0;
      halo_nelems [j] = 0;
      rcv_mem [j] = NULL;
      snd_mem [j] = NULL;
   }

   err = halo_setup (rank,
      tile_y_dim, tile_x_dim, matrix_y_dim, matrix_x_dim,
      &matrix_y, &matrix_y_above, &matrix_y_below,
      &matrix_x, &matrix_x_left,  &matrix_x_right,
      halo_ranks, halo_nelems, rcv_mem, snd_mem);

   fprintf (stderr, "MPI rank %d: halo setup:\n"
      "tile_y_dim %u tile_x_dim %u matrix_y_dim %hu matrix_x_dim %hu\n"
      "matrix_y %hu matrix_y_above %hu matrix_y_below %hu\n"
      "matrix_x %hu matrix_x_left  %hu matrix_x_right %hu\n"
      "halo_ranks\n"
      "%5d %5d %5d\n"
      "%5d       %5d\n"
      "%5d %5d %5d\n"
      "halo_nelems\n"
      "%5d %5d %5d\n"
      "%5d       %5d\n"
      "%5d %5d %5d\n",
      rank,
      tile_y_dim, tile_x_dim, matrix_y_dim, matrix_x_dim,
      matrix_y, matrix_y_above, matrix_y_below,
      matrix_x, matrix_x_left,  matrix_x_right,
      halo_ranks [UL  ], halo_ranks [ABOVE], halo_ranks [UR   ],
      halo_ranks [LEFT],                     halo_ranks [RIGHT],
      halo_ranks [LL  ], halo_ranks [BELOW], halo_ranks [LR   ],
      halo_nelems[UL  ], halo_nelems[ABOVE], halo_nelems[UR   ],
      halo_nelems[LEFT],                     halo_nelems[RIGHT],
      halo_nelems[LL  ], halo_nelems[BELOW], halo_nelems[LR   ]);


   // Create new HDF5 file using default properties
   file_id = H5Fcreate (fpath, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

   // Create dataspace for dataset
   dims[0] = steps; 
   dims[1] = tile_y_dim;   // rows
   dims[2] = tile_x_dim;   // columns
/*
   fprintf (stderr, "MPI rank %d: about to call H5Screate_simple to create dataspace for dataset\n", rank);
*/
   dataspace_id = H5Screate_simple (N_DIMS, dims, NULL);

   // Create property list for dataset
   hid_t plist_id = H5Pcreate (H5P_DATASET_CREATE);

   // Dataset must be chunked for compression
#define N_CHUNK_DIMS 3
   hsize_t cdims[N_CHUNK_DIMS] = { 1, tile_y_dim, tile_x_dim };
   status = H5Pset_chunk (plist_id, N_CHUNK_DIMS, cdims);

   // Compress the dataset with szip or zlib/DEFLATE
   if (USE_SZIP) {
      // Set szip parameters
      uint32 szip_options_mask = H5_SZIP_NN_OPTION_MASK;
      uint32 szip_pixels_per_block = 16;
      status = H5Pset_szip (plist_id, szip_options_mask, szip_pixels_per_block);
   }
   else {
      // Set zlib / DEFLATE compression level
      status = H5Pset_deflate (plist_id, ZLIB_COMPRESSION_LEVEL);
   }

   // Create dataset with datatype unsigned 16-bit integer, little-endian
   dataset_id = H5Dcreate2 (file_id, "/cell_data", H5T_STD_U16LE, dataspace_id,
      H5P_DEFAULT, plist_id, H5P_DEFAULT);

   // Create dataset attributes
   err = create_u32le_attr (rank, dataset_id, "rows",       tile_y_dim);
   err = create_u32le_attr (rank, dataset_id, "columns",    tile_x_dim);
   err = create_u32le_attr (rank, dataset_id, "time-steps", steps);
   err = create_u16le_attr (rank, dataset_id, "neighborhood-type", neighborhood_type);
   err = create_u16le_attr (rank, dataset_id, "V",  V);
   err = create_u16le_attr (rank, dataset_id, "g",  g);
   err = create_u16le_attr (rank, dataset_id, "k1", k1);
   err = create_u16le_attr (rank, dataset_id, "k2", k2);


   // Allocate memory for tile-without-halo

   uint16 *tile_mem = NULL;
   tile_mem = (uint16 *) calloc ((size_t) (tile_y_dim * tile_x_dim), sizeof(uint16));
   if (NULL == tile_mem) {
      fprintf (stderr, "MPI rank %d: calloc () failed for tile-without-halo\n",
         rank);
      exit (11);
   }

   // Allocate memory for tile and for halo around it.

   // Dimensions of tile-plus-halo
   // !!! Check that adding halo doesn't overflow uint32 dims
   uint32 tile_plus_halo_y_dim = 2 + tile_y_dim + 2;  // 2 rows on the top, 2 on the bottom
   uint32 tile_plus_halo_x_dim = 2 + tile_x_dim + 2;  // 2 columns on the left, 2 on the right

   uint16 *tile_plus_halo_mem = NULL;
   tile_plus_halo_mem = (uint16 *) calloc (
      (size_t) (tile_plus_halo_y_dim * tile_plus_halo_x_dim), sizeof(uint16) );
   if (NULL == tile_plus_halo_mem) {
      fprintf (stderr, "MPI rank %d: calloc () failed for tile-plus-halo\n",
         rank);
      exit (13);
   }


   // For step 0, fill tile-without-halo with random values in range [0, V]
   err = gen_random_uint16_0_to_V (rank, V,
      (size_t) (tile_y_dim * tile_x_dim), tile_mem);


   // Write step 0, stored in tile-without-halo, to HDF5 file
/*
   fprintf (stderr, "MPI rank %d: step 0: about to write to HDF5 file\n", rank);
*/
   err = write_step (rank, dataset_id, 0, tile_y_dim, tile_x_dim, tile_mem);


   // Wait for all MPI ranks to reach this point before continuing
   MPI_Barrier (MPI_COMM_WORLD);
/*
   fprintf (stderr, "MPI rank %d: step 0: barrier\n", rank);
*/


   // Let the cellular automaton evolve!
   // For each step [1, 2, ... steps], do the following:
   // 1. Copy previous step data from tile-without-halo to tile-plus-halo.
   // 2. Obtain previous step halo data from adjacent tiles by communicating
   //    with other MPI ranks.  Write halo data to tile-plus-halo.
   // 3. Wait for all MPI ranks to reach this point before continuing.
   // 4. Use previous step data stored in tile-plus-halo to compute new
   //    values.  Write new values to tile-without-halo.
   // 5. Write tile-without-halo to HDF5 file.
   // 6. Wait for all MPI ranks to reach this point before continuing.


   uint32 row = 0, col = 0;
   uint32 tick = 1;

   for (tick = 1; tick < steps; tick++) {

      // 1. Copy previous step data from tile-without-halo to tile-plus-halo
/*
      fprintf (stderr, "MPI rank %d: step %u: about to copy previous values to"
         " tile-plus-halo\n", rank, tick);
*/
      for (row = 0; row < tile_y_dim; row++) {
         for (col = 0; col < tile_x_dim; col++) {
            tile_plus_halo_mem [ (2+row) * tile_plus_halo_x_dim + (2+col) ]  =
                      tile_mem [    row  *           tile_x_dim +    col  ];
         }
      }

      // 2. Exchange halo data with MPI ranks processing adjacent tiles
/*
      fprintf (stderr, "MPI rank %d: step %u: about to exchange halo data\n",
         rank, tick);
*/
      err = halo_exchange (rank,
         tile_y_dim, tile_x_dim, tile_mem,
         tile_plus_halo_y_dim, tile_plus_halo_x_dim, tile_plus_halo_mem,
         halo_ranks, halo_nelems, rcv_mem, snd_mem);

      // 3. Wait for all MPI ranks to reach this point before continuing

      MPI_Barrier (MPI_COMM_WORLD);
/*
      fprintf (stderr, "MPI rank %d: step %u: barrier\n", rank, tick);
*/

      // 4. Use previous step data stored in tile-plus-halo to compute new
      //    values.  Write new values to tile-without-halo
/*
      fprintf (stderr, "MPI rank %d: step %u: about to compute new values\n",
         rank, tick);
*/
      err = evolve_hodge_podge (rank, tick, neighborhood_type, V, g, k1, k2,
         tile_plus_halo_y_dim, tile_plus_halo_x_dim, tile_plus_halo_mem,
         tile_y_dim, tile_x_dim, tile_mem);

      // 5. Write tile-without-halo to HDF5 file
/*
      fprintf (stderr, "MPI rank %d: step %u: about to write to HDF5 file\n",
         rank, tick);
*/
      err = write_step (rank, dataset_id, tick, tile_y_dim, tile_x_dim,
         tile_mem);

      // 6. Wait for all MPI ranks to reach this point before continuing

      MPI_Barrier (MPI_COMM_WORLD);
/*
      fprintf (stderr, "MPI rank %d: step %u: barrier\n", rank, tick);
*/
   }


   // Clean up HDF5 resources
   status = H5Dclose (dataset_id);
   status = H5Sclose (dataspace_id);
   status = H5Pclose (plist_id);
   status = H5Fclose (file_id);


   // Wait for all MPI ranks to reach this point before continuing
   MPI_Barrier (MPI_COMM_WORLD);

   MPI_Finalize ();


   exit (0);
}


/*------------------------------------------------------------------------------
halo_setup

Set up halo exchange
------------------------------------------------------------------------------*/
int halo_setup (int rank,
   uint16 tile_y_dim, uint16 tile_x_dim, uint16 matrix_y_dim, uint16 matrix_x_dim,
   uint16 *matrix_y, uint16 *matrix_y_above, uint16 *matrix_y_below,
   uint16 *matrix_x, uint16 *matrix_x_left,  uint16 *matrix_x_right,
   int *halo_ranks, int *halo_nelems, uint16 *rcv_mem[], uint16 *snd_mem[])
{
   // Given MPI rank, figure out coordinates of tile in matrix
   // Example: 4 x 3 matrix:
   //  0  1  2  3
   //  4  5  6  7
   //  8  9 10 11

   *matrix_y = rank / matrix_x_dim;  // integer arithmetic with truncation
   *matrix_x = rank % matrix_x_dim;

   // Figure out which MPI ranks are halo exchange neighbors.
   // Right wraps around to left and bottom wraps around to top

   *matrix_y_above = (*matrix_y + matrix_y_dim - 1) % matrix_y_dim;
   *matrix_y_below = (*matrix_y                + 1) % matrix_y_dim;

   *matrix_x_left  = (*matrix_x + matrix_x_dim - 1) % matrix_x_dim;
   *matrix_x_right = (*matrix_x                + 1) % matrix_x_dim;

   halo_ranks [ABOVE] = *matrix_y_above * matrix_x_dim + *matrix_x;
   halo_ranks [BELOW] = *matrix_y_below * matrix_x_dim + *matrix_x;
   halo_ranks [LEFT ] = *matrix_y       * matrix_x_dim + *matrix_x_left;
   halo_ranks [RIGHT] = *matrix_y       * matrix_x_dim + *matrix_x_right;
   halo_ranks [UL   ] = *matrix_y_above * matrix_x_dim + *matrix_x_left;
   halo_ranks [UR   ] = *matrix_y_above * matrix_x_dim + *matrix_x_right;
   halo_ranks [LL   ] = *matrix_y_below * matrix_x_dim + *matrix_x_left;
   halo_ranks [LR   ] = *matrix_y_below * matrix_x_dim + *matrix_x_right;

   // Figure out how many data elements are in each piece of halo exchange
   // memory

   halo_nelems[ABOVE] = halo_nelems[BELOW] = 2 * tile_x_dim;
   halo_nelems[LEFT ] = halo_nelems[RIGHT] = tile_y_dim * 2;
   halo_nelems[UL] = halo_nelems[UR] = halo_nelems[LL] = halo_nelems[LR] = 2*2;

   // Allocate halo exchange memory

   size_t n = 0;

   for (n = 0; n < N_HALO_NEIGHBORS; n++) {
      rcv_mem[n] = (uint16 *) calloc ((size_t) halo_nelems[directions[n]],
         sizeof(uint16));
      if (NULL == rcv_mem[n]) {
         fprintf (stderr, "MPI rank %d: calloc () failed for rcv_mem[%d]\n", rank, n);
         exit (21);
      }
      snd_mem[n] = (uint16 *) calloc ((size_t) halo_nelems[reverse_directions[n]],
         sizeof(uint16));
      if (NULL == snd_mem[n]) {
         fprintf (stderr, "MPI rank %d: calloc () failed for snd_mem[%d]\n", rank, n);
         exit (23);
      }
   }

   return 0;
}


/*------------------------------------------------------------------------------
halo_exchange

Exchange halo data with MPI ranks processing adjacent tiles
------------------------------------------------------------------------------*/
int halo_exchange (int rank,
   uint16 tile_y_dim, uint16 tile_x_dim,
   uint16 *tile_mem,
   uint16 tile_plus_halo_y_dim, uint16 tile_plus_halo_x_dim,
   uint16 *tile_plus_halo_mem,
   int *halo_ranks, int *halo_nelems, uint16 *rcv_mem[], uint16 *snd_mem[])
{
   size_t row = 0, col = 0, idx = 0;

   // Copy this rank's halo data into halo buffers

   // above
   idx = 0;
   for (row = 0; row < 2; row++) {
      for (col = 0; col < tile_x_dim; col++) {
         snd_mem [ABOVE] [idx++] = tile_mem [row * tile_x_dim + col];
      }
   }

   // below
   idx = 0;
   for (row = tile_y_dim - 2; row < tile_y_dim; row++) {
      for (col = 0; col < tile_x_dim; col++) {
         snd_mem [BELOW] [idx++] = tile_mem [row * tile_x_dim + col];
      }
   }

   // left
   idx = 0;
   for (row = 0; row < tile_y_dim; row++) {
      for (col = 0; col < 2; col++) {
         snd_mem [LEFT ] [idx++] = tile_mem [row * tile_x_dim + col];
      }
   }

   // right
   idx = 0;
   for (row = 0; row < tile_y_dim; row++) {
      for (col = tile_x_dim - 2; col < tile_x_dim; col++) {
         snd_mem [RIGHT] [idx++] = tile_mem [row * tile_x_dim + col];
      }
   }

   // upper left (ul)
   idx = 0;
   for (row = 0; row < 2; row++) {
      for (col = 0; col < 2; col++) {
         snd_mem [UL] [idx++] = tile_mem [row * tile_x_dim + col];
      }
   }

   // upper right (ur)
   idx = 0;
   for (row = 0; row < 2; row++) {
      for (col = tile_x_dim - 2; col < tile_x_dim; col++) {
         snd_mem [UR] [idx++] = tile_mem [row * tile_x_dim + col];
      }
   }

   // lower left (ll)
   idx = 0;
   for (row = tile_y_dim - 2; row < tile_y_dim; row++) {
      for (col = 0; col < 2; col++) {
         snd_mem [LL] [idx++] = tile_mem [row * tile_x_dim + col];
      }
   }

   // lower right (lr)
   idx = 0;
   for (row = tile_y_dim - 2; row < tile_y_dim; row++) {
      for (col = tile_x_dim - 2; col < tile_x_dim; col++) {
         snd_mem [LR] [idx++] = tile_mem [row * tile_x_dim + col];
      }
   }

#define N_REQ (2 * N_HALO_NEIGHBORS)   // a receive and a send for each neighbor
   MPI_Request req  [N_REQ];
   MPI_Status  stat [N_REQ];

   int n = 0;
   int err = 0;

   // RECEIVE - Make non-blocking Irecv calls
   // Tag is direction to sending rank from this receiving rank's point of view
   // int MPI_Irecv(void *buf, int count, MPI_Datatype datatype,
   //    int source, int tag, MPI_Comm comm, MPI_Request *request)
   for (n = 0; n < N_HALO_NEIGHBORS; n++) {
      err = MPI_Irecv (rcv_mem[n], halo_nelems[n], MPI_UNSIGNED_SHORT,
         halo_ranks[n], directions[n], MPI_COMM_WORLD,
         &req[ n ]);
         // req [0, N_HALO_NEIGHBORS-1] are Irecv
   }

   // ??? After issuing non-blocking receive requests, can we proceed
   // ??? immediately to issue non-blocking send requests?

   // SEND - Make non-blocking Isend calls
   // Tag is direction from this sending rank to the receiving rank
   // int MPI_Isend(const void *buf, int count, MPI_Datatype datatype,
   //    int dest, int tag, MPI_Comm comm, MPI_Request *request)
   for (n = 0; n < N_HALO_NEIGHBORS; n++) {
      err = MPI_Isend (snd_mem[n], halo_nelems[n], MPI_UNSIGNED_SHORT,
         halo_ranks[n], reverse_directions[n], MPI_COMM_WORLD,
         &req[ N_HALO_NEIGHBORS + n ]);
         // req [N_HALO_NEIGHBORS, 2*N_HALO_NEIGHBORS-1] are Isend
   }

   // Wait for all of the Irecv and Isend calls to finish
   // int MPI_Waitall(int count, MPI_Request array_of_requests[],
   //    MPI_Status *array_of_statuses)
   err = MPI_Waitall(N_REQ, req, stat);

   // Copy received halo data into tile-plus-halo

   // above
   idx = 0;
   for (row = 0; row < 2; row++) {
      for (col = 2; col < tile_plus_halo_x_dim - 2; col++) {
         tile_plus_halo_mem [row * tile_plus_halo_x_dim + col] = rcv_mem [ABOVE] [idx++];
      }
   }

   // below
   idx = 0;
   for (row = tile_plus_halo_y_dim - 2; row < tile_plus_halo_y_dim; row++) {
      for (col = 2; col < tile_plus_halo_x_dim - 2; col++) {
         tile_plus_halo_mem [row * tile_plus_halo_x_dim + col] = rcv_mem [BELOW] [idx++];
      }
   }

   // left
   idx = 0;
   for (row = 2; row < tile_plus_halo_y_dim - 2; row++) {
      for (col = 0; col < 2; col++) {
         tile_plus_halo_mem [row * tile_plus_halo_x_dim + col] = rcv_mem [LEFT ] [idx++];
      }
   }

   // right
   idx = 0;
   for (row = 2; row < tile_plus_halo_y_dim - 2; row++) {
      for (col = tile_plus_halo_x_dim - 2; col < tile_plus_halo_x_dim; col++) {
         tile_plus_halo_mem [row * tile_plus_halo_x_dim + col] = rcv_mem [RIGHT] [idx++];
      }
   }

   // upper left (ul)
   idx = 0;
   for (row = 0; row < 2; row++) {
      for (col = 0; col < 2; col++) {
         tile_plus_halo_mem [row * tile_plus_halo_x_dim + col] = rcv_mem [UL] [idx++];
      }
   }

   // upper right (ur)
   idx = 0;
   for (row = 0; row < 2; row++) {
      for (col = tile_plus_halo_x_dim - 2; col < tile_plus_halo_x_dim; col++) {
         tile_plus_halo_mem [row * tile_plus_halo_x_dim + col] = rcv_mem [UR] [idx++];
      }
   }

   // lower left (ll)
   idx = 0;
   for (row = tile_plus_halo_y_dim - 2; row < tile_plus_halo_y_dim; row++) {
      for (col = 0; col < 2; col++) {
         tile_plus_halo_mem [row * tile_plus_halo_x_dim + col] = rcv_mem [LL] [idx++];
      }
   }

   // lower right (lr)
   idx = 0;
   for (row = tile_plus_halo_y_dim - 2; row < tile_plus_halo_y_dim; row++) {
      for (col = tile_plus_halo_x_dim - 2; col < tile_plus_halo_x_dim; col++) {
         tile_plus_halo_mem [row * tile_plus_halo_x_dim + col] = rcv_mem [LR] [idx++];
      }
   }

   return 0;
}


/*------------------------------------------------------------------------------
create_u16le_attr

Create an unsigned 16-bit little-endian scalar attribute in the HDF5 dataset
------------------------------------------------------------------------------*/

static int create_u16le_attr (int rank, hid_t dset_id, const char *att_name,
   uint16 att_val)
{
   hid_t att_dspace_id;  // attribute dataspace identifier
   hid_t att_id;         // attribute identifier
   herr_t status;

   // Create dataspace for scalar attribute
   att_dspace_id = H5Screate (H5S_SCALAR);

   // Create scalar attribute
   att_id = H5Acreate2 (dset_id, att_name, H5T_STD_U16LE, att_dspace_id,
      H5P_DEFAULT, H5P_DEFAULT);

   // Write scalar attribute
   status = H5Awrite (att_id, H5T_NATIVE_USHORT, &att_val);

   // Close attribute dataspace
   status = H5Sclose (att_dspace_id);

   // Close attribute
   status = H5Aclose (att_id);

   return 0;
}


/*------------------------------------------------------------------------------
create_u32le_attr

Create an unsigned 32-bit little-endian scalar attribute in the HDF5 dataset
------------------------------------------------------------------------------*/

static int create_u32le_attr (int rank, hid_t dset_id, const char *att_name,
   uint32 att_val)
{
   hid_t att_dspace_id;  // attribute dataspace identifier
   hid_t att_id;         // attribute identifier
   herr_t status;

   // Create dataspace for scalar attribute
   att_dspace_id = H5Screate (H5S_SCALAR);

   // Create scalar attribute
   att_id = H5Acreate2 (dset_id, att_name, H5T_STD_U32LE, att_dspace_id,
      H5P_DEFAULT, H5P_DEFAULT);

   // Write scalar attribute
   status = H5Awrite (att_id, H5T_NATIVE_UINT, &att_val);  // ??? is NATIVE_UINT correct ???

   // Close attribute dataspace
   status = H5Sclose (att_dspace_id);

   // Close attribute
   status = H5Aclose (att_id);

   return 0;
}


/*------------------------------------------------------------------------------
gen_random_uint16_0_to_V

Fill `mem' with pseudo-random unsigned 16-bit values in the range [0, V].
Take care to avoid bias
------------------------------------------------------------------------------*/

static int gen_random_uint16_0_to_V (int rank, uint16 V, size_t len,
   uint16 *mem)
{
   // Allocate a temporary buffer for random numbers.  We'll generate a bunch
   // of them and use them (or discard them) one by one, then generate more
   uint16 *buf = NULL;
#define NUM_ELEMS 1024
   buf = (uint16 *) calloc ((size_t) NUM_ELEMS, sizeof(uint16));
   if (NULL == buf) {
      fprintf (stderr, "calloc () failed\n");
      exit (29);
   }

#define MAX_UINT16_VALUE 65535
   uint16 max_usable_random =
        (MAX_UINT16_VALUE == V)
      ?  MAX_UINT16_VALUE
      : (MAX_UINT16_VALUE / (V + 1)) * (V + 1) - 1;  // integer div

   size_t remaining_randoms = 0;
   size_t next_cell = 0;

   while (next_cell < len) {
      // If buffer of random numbers is empty, generate a bunch
      if (0 == remaining_randoms) {
         // RAND_bytes () produces unsigned chars (uint8).  We need 2*NUM_ELEMS
         // of them to fill the uint16 temporary buffer
         if (1 != RAND_bytes ((uint8 *) buf, 2 * NUM_ELEMS)) {  // 1 is success
            fprintf (stderr, "RAND_bytes () failed\n");
            exit (31);
         }
         remaining_randoms = NUM_ELEMS;
      }
      // In order to avoid bias, we can only use random numbers in the range
      // [0, N*(V+1)-1] modulo V+1.
      // We want random numbers in the range [0, V], a set of V+1 numbers.
      // We have a buffer full of random numbers in the range [0, 65535].
      // We can use values [0, V] straightaway.
      // We can also use [V+1, 2V+1] modulo V+1.
      // And so on with [2V+2, 3V+2], [3V+3, 4V+3], ..., up to
      // [(N-1)V+(N-1), NV+(N-1)].  Note that NV+(N-1) is the same as N*(V+1)-1.
      if (buf [remaining_randoms-1] <= max_usable_random) {
         mem [next_cell] = buf [remaining_randoms-1] % (V + 1);
         ++next_cell;
      }
      --remaining_randoms;
   }

   // Clean up random number buffer
   free (buf);  buf = NULL;

   return 0;
}


/*------------------------------------------------------------------------------
evolve_hodge_podge

Compute next step of "hodge podge machine"
------------------------------------------------------------------------------*/

static int evolve_hodge_podge (int rank, uint32 step, uint8 neighborhood_type,
   uint16 V, uint16 g, uint16 k1, uint16 k2,
   uint32 tile_plus_halo_y_dim, uint32 tile_plus_halo_x_dim,
   uint16 *tile_plus_halo_mem,
   uint32 tile_y_dim, uint32 tile_x_dim,
   uint16 *tile_mem)
{
   uint32 row = 0, col = 0;
   uint16 oldval = 0, newval = 0;

   for (row = 0; row < tile_y_dim; row++) {

      for (col = 0; col < tile_x_dim; col++) {

         oldval = tile_plus_halo_mem [ (2+row) * tile_plus_halo_x_dim + (2+col) ];

/*
         fprintf (stderr, "MPI rank %d: step %u: row %u col %u oldval %hu\n",
            rank, step, row, col, oldval);
*/

         // If cell is "ill" (V), it miraculously becomes "healthy" (0)
         if (oldval == V) {
            newval = 0;
            tile_mem [row * tile_x_dim + col] = newval;
         }

         else {

            // Cell is either "healthy" (0) or "infected" (0 < state < V)

            // Count "infected" neighbors (Iij) and "ill" neighbors (Kij)
            // and compute sum of states of "infected" neighbors (Sij)

            uint32 Sij = 0;
            uint16 Iij = 0, Kij = 0, neighbor_state = 0;
            uint8 jj = 0;
            sint8 xoff = 0, yoff = 0;

            for (jj = 0; jj < num_neighbors [EXT_MOORE]; jj++) {
               yoff = ext_moore_y [jj];
               xoff = ext_moore_x [jj];
               neighbor_state = tile_plus_halo_mem [(2+row+yoff) * tile_plus_halo_x_dim + (2+col+xoff)];
               if (0 < neighbor_state && neighbor_state < V) {
                  Sij += (uint32) neighbor_state;
                  ++Iij;
               }
               if (neighbor_state == V) {
                  ++Kij;
               }
            }

            // If cell is "healthy" (0), it becomes "infected" to a degree
            // depending on the number of "ill" neighbors (Kij), on the
            // number of "infected" neighbors (Iij), and on the parameters
            // k1 and k2

            if (0 == oldval) {

               // We're intentionally doing integer division with truncation here
               newval = (Kij / k1) + (Iij / k2);
               tile_mem [row * tile_x_dim + col] = newval;
            }

            // If cell is "infected" (0 < value < V), its new "ill" or
            // "infected" state depends on the sum of the degrees of
            // "infection" of the cell's neighbors (Sij), on the number
            // of "infected" neighbors (Iij), and on the parameter g

            else if (0 < oldval && oldval < V) {

               // We're intentionally doing integer division with truncation here
               uint32 aa = (Sij / (uint32) Iij) + (uint32) g;
               uint32 bb = (uint32) V;
               newval = (uint16) (aa < bb ? aa : bb);  // min (aa, bb)
               tile_mem [row * tile_x_dim + col] = newval;
            }

            else {
               // We should never reach this point!  Something is badly wrong!
               fprintf (stderr, "MPI rank %d: oldval out of expected range"
                  " [0, V]: something badly wrong: quitting\n", rank);
               exit (41);
            }

         }
      }
   }

   return 0;
}


/*------------------------------------------------------------------------------
write_step

Write step tile data to HDF5 file
------------------------------------------------------------------------------*/

static int write_step (int rank, hid_t dset_id, uint32 step,
   uint32 y_dim, uint32 x_dim, uint16 *step_mem)
{
   herr_t status = 0;

   // Specify size and shape of subset of HDF5 dataspace to write

   hsize_t offset [N_DIMS];
   hsize_t count  [N_DIMS];
   hsize_t stride [N_DIMS];
   hsize_t block  [N_DIMS];

   offset [0] = step;
   offset [1] = 0;
   offset [2] = 0;

   count  [0] = 1;  // steps
   count  [1] = y_dim;  // rows
   count  [2] = x_dim;  // columns

   stride [0] = 1;
   stride [1] = 1;
   stride [2] = 1;

   block  [0] = 1;
   block  [1] = 1;
   block  [2] = 1;

   // Create memory space with size of subset.  Get file dataspace and select
   // subset of file dataspace

   hsize_t dimsm [N_DIMS];
   dimsm [0] = 1;      // steps
   dimsm [1] = y_dim;  // rows
   dimsm [2] = x_dim;  // columns

/*
   fprintf (stderr, "MPI rank %d: about to call H5Screate_simple to create memory space with size of subset\n", rank);
*/
   hid_t memspace_id = H5Screate_simple (N_DIMS, dimsm, NULL); 

   hid_t sub_dataspace_id = H5Dget_space (dset_id);
   status = H5Sselect_hyperslab (sub_dataspace_id, H5S_SELECT_SET, offset,
      stride, count, block);

   // Copy step to HDF5 file.  Note that while we're writing native-endian
   // 16-bit values from memory to the HDF5 file, the datatype of the dataset
   // in the file is little-endian, and the HDF5 library will take care of
   // byte-swapping if necessary

/*
   fprintf (stderr, "MPI rank %d: step %d: about to write to HDF5 file\n",
      rank, step);
*/
   status = H5Dwrite (dset_id, H5T_NATIVE_USHORT, memspace_id,
      sub_dataspace_id, H5P_DEFAULT, step_mem);

   // End access to dataspaces
   status = H5Sclose (sub_dataspace_id);
   status = H5Sclose (memspace_id);
    
   return 0;
}


