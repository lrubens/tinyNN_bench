CC=g++
CFLAGS=-O3 -mavx2 -mavx512f -march=native -ffast-math -funroll-loops -mfma
DEPS = common.h
OBJ = mm.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) -I./

bench: $(OBJ)
	$(CC) -o $@ $^ bench.c $(CFLAGS)

clean:
	rm -rf *.o bench


