CC=riscv64-unknown-linux-gnu-g++
CXXFLAGS=-march=rv64gcv0p7_zfh_xtheadc -mabi=lp64d -mtune=c906 -DC906=1 -static

pack_a: test_pack_A.cpp
	$(CC) $^ -O3  -o $@ $(CXXFLAGS)

run: pack_a
	qemu-riscv64 -cpu c906fdv ./pack_a

pack_b: test_pack_B_handle.cpp
	$(CC) $^ -O3  -o $@ $(CXXFLAGS)

runb: pack_b
	qemu-riscv64 -cpu c906fdv ./pack_b

gemm:
	$(CC) test_gemm_transB_packed_tile.cpp -O3  -c -o $@ $(CXXFLAGS)

transpose:
	$(CC) test_transpose_unpack_output_tile.cpp -O3  -c -o $@ $(CXXFLAGS)

clean:
	rm -rf pack_a
	rm -rf pack_b

# clang armcommon_tinycv_rotate_fp16.c armcommon_tinycv_rotate_ui8.c main_fp16.c -O3  -o mat_fp16