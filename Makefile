WHOLE_LINK = -dlto -arch=sm_86
# WHOLE_LINK =-dlto -arch=sm_75#

LDLIBS = -lcrypto \
		 -L/data/cuda/cuda-11.8/lib64 -lcudart -lstdc++ -lm $(WHOLE_LINK)
CC = nvcc -ccbin g++ --expt-relaxed-constexpr #--ptxas-options=-v
# CC = nvcc -ccbin g++ -Xcompiler -rdynamic -lineinfo
CUFLAGS = -O3 -m64 -dc $(WHOLE_LINK) #--maxrregcount=70
DIR_OBJ = ./obj
DIR_BIN = ./bin

SOURCES = common.cu address.cu rng.cu wots.cu utils.cu fors.cu sign.cu \
		fips202.cu gpu_fips202.cu hash_shake256.cu thash_shake256.cu \
		sha256.cu gpu_sha256.cu hash_sha256.cu thash_sha256.cu

SOURCES_BM = common.cu address.cu rng.cu wots.cu utils.cu fors.cu sign.cu \
		fips202.cu gpu_fips202.cu hash_shake256.cu thash_shake256.cu \
		sha256.cu gpu_sha256.cu hash_sha256.cu thash_sha256.cu bm_ap.cu

HEADERS = all_option.h params.h common.h\
		address.h rng.h wots.h utils.h fors.h api.h hash.h thash.h \
		fips202.h sha256.h

SHA256 = bin/sha256
WOTS = bin/wots
FORS = bin/fors
SPX = bin/spx # serial
AP_SPX = bin/ap_spx # algorithmic parallelism
DP_SPX = bin/dp_spx # data parallelism
HP_SPX = bin/hp_spx # hybrid parallelism

BENCHMARK = bin/benchmark
AP_BENCHMARK = bin/ap_benchmark
DP_BENCHMARK = bin/dp_benchmark
HP_BENCHMARK = bin/hp_benchmark

sha256: $(SHA256:=.exec)
wots: $(WOTS:=.exec)
fors: $(FORS:=.exec)
spx: $(SPX:=.exec)
ap: $(AP_SPX:=.exec)
dp: $(DP_SPX:=.exec)
hp: $(HP_SPX:=.exec)

benchmark: $(BENCHMARK)
ap_benchmark: $(AP_BENCHMARK)
dp_benchmark: $(DP_BENCHMARK)
hp_benchmark: $(HP_BENCHMARK)
# hp_benchmark: $(HP_BENCHMARK)

OBJECTS = $(patsubst %.cu, $(DIR_OBJ)/%.o, $(SOURCES))
OBJECTS_BM = $(patsubst %.cu, $(DIR_OBJ)/%.o, $(SOURCES_BM))

.PRECIOUS: $(DIR_OBJ)/%.o # Keep these intermediate files
.PRECIOUS: $(DIR_BIN)/%.o # Keep these intermediate files

# generation of intermediate file
# default
$(DIR_OBJ)/%.o: %.cu $(HEADERS)
	@mkdir -p $(DIR_OBJ)
	$(CC) $(CUFLAGS) -c -o "$@" "$<"

test/%.o: test/%.cu
	$(CC) $(CUFLAGS) -c -o "$@" "$<"

bin/%: test/%.o $(OBJECTS)
	mkdir -p bin
	$(CC) -O3 -o $@ $^ $(LDLIBS)

bin/%.exec: bin/%
	@$<

# ------------------------------------------
# ------------------------------------------
# for fast tests
# ------------------------------------------
# ------------------------------------------
TARGETS = \
    bin/bm-sha256-128s bin/bm-sha256-128f \
    bin/bm-sha256-192s bin/bm-sha256-192f \
    bin/bm-sha256-256s bin/bm-sha256-256f \
	bin/bm-shake256-128s bin/bm-shake256-128f \
    bin/bm-shake256-192s bin/bm-shake256-192f \
    bin/bm-shake256-256s bin/bm-shake256-256f

benchmark_all: $(TARGETS)

record:
	for x in $(TARGETS); \
	do \
	echo $$x >> 1.txt; \
	./$$x >> 1.txt; \
	done

OBJECTS-sha256-128s = $(patsubst %.cu, $(DIR_OBJ)/%_sha256-128s.o, $(SOURCES_BM))
OBJECTS-sha256-128f = $(patsubst %.cu, $(DIR_OBJ)/%_sha256-128f.o, $(SOURCES_BM))
OBJECTS-sha256-192s = $(patsubst %.cu, $(DIR_OBJ)/%_sha256-192s.o, $(SOURCES_BM))
OBJECTS-sha256-192f = $(patsubst %.cu, $(DIR_OBJ)/%_sha256-192f.o, $(SOURCES_BM))
OBJECTS-sha256-256s = $(patsubst %.cu, $(DIR_OBJ)/%_sha256-256s.o, $(SOURCES_BM))
OBJECTS-sha256-256f = $(patsubst %.cu, $(DIR_OBJ)/%_sha256-256f.o, $(SOURCES_BM))

OBJECTS-shake256-128s = $(patsubst %.cu, $(DIR_OBJ)/%_shake256-128s.o, $(SOURCES_BM))
OBJECTS-shake256-128f = $(patsubst %.cu, $(DIR_OBJ)/%_shake256-128f.o, $(SOURCES_BM))
OBJECTS-shake256-192s = $(patsubst %.cu, $(DIR_OBJ)/%_shake256-192s.o, $(SOURCES_BM))
OBJECTS-shake256-192f = $(patsubst %.cu, $(DIR_OBJ)/%_shake256-192f.o, $(SOURCES_BM))
OBJECTS-shake256-256s = $(patsubst %.cu, $(DIR_OBJ)/%_shake256-256s.o, $(SOURCES_BM))
OBJECTS-shake256-256f = $(patsubst %.cu, $(DIR_OBJ)/%_shake256-256f.o, $(SOURCES_BM))

$(DIR_OBJ)/%_sha256-128s.o: %.cu $(HEADERS)
	@mkdir -p $(DIR_OBJ)
	$(CC) -DVARIANT -DSHA256 -DSPX_128S -DSIMPLE $(CUFLAGS) -c -o "$@" "$<"

$(DIR_OBJ)/%_sha256-128f.o: %.cu $(HEADERS)
	@mkdir -p $(DIR_OBJ)
	$(CC) -DVARIANT -DSHA256 -DSPX_128F -DSIMPLE $(CUFLAGS) -c -o "$@" "$<"

$(DIR_OBJ)/%_sha256-192s.o: %.cu $(HEADERS)
	@mkdir -p $(DIR_OBJ)
	$(CC) -DVARIANT -DSHA256 -DSPX_192S -DSIMPLE $(CUFLAGS) -c -o "$@" "$<"

$(DIR_OBJ)/%_sha256-192f.o: %.cu $(HEADERS)
	@mkdir -p $(DIR_OBJ)
	$(CC) -DVARIANT -DSHA256 -DSPX_192F -DSIMPLE $(CUFLAGS) -c -o "$@" "$<"

$(DIR_OBJ)/%_sha256-256s.o: %.cu $(HEADERS)
	@mkdir -p $(DIR_OBJ)
	$(CC) -DVARIANT -DSHA256 -DSPX_256S -DSIMPLE $(CUFLAGS) -c -o "$@" "$<"

$(DIR_OBJ)/%_sha256-256f.o: %.cu $(HEADERS)
	@mkdir -p $(DIR_OBJ)
	$(CC) -DVARIANT -DSHA256 -DSPX_256F -DSIMPLE $(CUFLAGS) -c -o "$@" "$<"

$(DIR_OBJ)/%_shake256-128s.o: %.cu $(HEADERS)
	@mkdir -p $(DIR_OBJ)
	$(CC) -DVARIANT -DSHAKE256 -DSPX_128S -DSIMPLE $(CUFLAGS) -c -o "$@" "$<"

$(DIR_OBJ)/%_shake256-128f.o: %.cu $(HEADERS)
	@mkdir -p $(DIR_OBJ)
	$(CC) -DVARIANT -DSHAKE256 -DSPX_128F -DSIMPLE $(CUFLAGS) -c -o "$@" "$<"

$(DIR_OBJ)/%_shake256-192s.o: %.cu $(HEADERS)
	@mkdir -p $(DIR_OBJ)
	$(CC) -DVARIANT -DSHAKE256 -DSPX_192S -DSIMPLE $(CUFLAGS) -c -o "$@" "$<"

$(DIR_OBJ)/%_shake256-192f.o: %.cu $(HEADERS)
	@mkdir -p $(DIR_OBJ)
	$(CC) -DVARIANT -DSHAKE256 -DSPX_192F -DSIMPLE $(CUFLAGS) -c -o "$@" "$<"

$(DIR_OBJ)/%_shake256-256s.o: %.cu $(HEADERS)
	@mkdir -p $(DIR_OBJ)
	$(CC) -DVARIANT -DSHAKE256 -DSPX_256S -DSIMPLE $(CUFLAGS) -c -o "$@" "$<"

$(DIR_OBJ)/%_shake256-256f.o: %.cu $(HEADERS)
	@mkdir -p $(DIR_OBJ)
	$(CC) -DVARIANT -DSHAKE256 -DSPX_256F -DSIMPLE $(CUFLAGS) -c -o "$@" "$<"

bin/bm-sha256-128s: $(OBJECTS-sha256-128s)
	$(CC) -O3 -o $@ $^ $(LDLIBS)
bin/bm-sha256-128f: $(OBJECTS-sha256-128f)
	$(CC) -O3 -o $@ $^ $(LDLIBS)
bin/bm-sha256-192s: $(OBJECTS-sha256-192s)
	$(CC) -O3 -o $@ $^ $(LDLIBS)
bin/bm-sha256-192f: $(OBJECTS-sha256-192f)
	$(CC) -O3 -o $@ $^ $(LDLIBS)
bin/bm-sha256-256s: $(OBJECTS-sha256-256s)
	$(CC) -O3 -o $@ $^ $(LDLIBS)
bin/bm-sha256-256f: $(OBJECTS-sha256-256f)
	$(CC) -O3 -o $@ $^ $(LDLIBS)

bin/bm-shake256-128s: $(OBJECTS-shake256-128s)
	$(CC) -O3 -o $@ $^ $(LDLIBS)
bin/bm-shake256-128f: $(OBJECTS-shake256-128f)
	$(CC) -O3 -o $@ $^ $(LDLIBS)
bin/bm-shake256-192s: $(OBJECTS-shake256-192s)
	$(CC) -O3 -o $@ $^ $(LDLIBS)
bin/bm-shake256-192f: $(OBJECTS-shake256-192f)
	$(CC) -O3 -o $@ $^ $(LDLIBS)
bin/bm-shake256-256s: $(OBJECTS-shake256-256s)
	$(CC) -O3 -o $@ $^ $(LDLIBS)
bin/bm-shake256-256f: $(OBJECTS-shake256-256f)
	$(CC) -O3 -o $@ $^ $(LDLIBS)

clean:
	-$(RM) $(TESTS)
	-$(RM) $(BENCHMARK)
	-$(RM) $(AP_BENCHMARK)
	-$(RM) $(HP_BENCHMARK)
	-$(RM) $(DP_BENCHMARK)
	-$(RM) $(DIR_OBJ)/*
	-$(RM) test/*.o

clean_all:
	make clean
	-$(RM) $(TARGETS)
