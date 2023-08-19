pack_a: test_pack_A.cpp
	clang++ $^ -O3  -o $@

run: pack_a
	./pack_a

pack_b: test_pack_B.cpp
	clang++ $^ -O3  -o $@

gemm: test_gemm_transB_packed_tile.cpp
	clang++ $^ -O3 -c -o $@

runb: pack_b
	./pack_b