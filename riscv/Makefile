CC=riscv64-unknown-linux-gnu-g++
CXXFLAGS=-march=rv64gcv0p7_zfh_xtheadc -mabi=lp64d -mtune=c906 -DC906=1 -static

pack_a: test_pack_A.cpp
	$(CC) $^ -O3  -o $@ $(CXXFLAGS)

run: pack_a
	qemu-riscv64 -cpu c906fdv ./pack_a

clean:
	rm -rf pack_a

# clang armcommon_tinycv_rotate_fp16.c armcommon_tinycv_rotate_ui8.c main_fp16.c -O3  -o mat_fp16