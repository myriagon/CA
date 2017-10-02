all: hodge-podge

hodge-podge: hodge-podge.c my-types.h neighborhoods.h
	mpicxx -o hodge-podge hodge-podge.c -lhdf5 -lhdf5_hl -lcrypto
	cp -p hodge-podge /shared/bin

clean:
	rm -f hodge-podge hodge-podge.o

