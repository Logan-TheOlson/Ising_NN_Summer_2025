// ising_csv_opt.cpp — Random-hit Metropolis, maximum speed with safe physics
// Same method & CSV as before. Optimizations:
// - Time-based, end-only progress (no mid-task atomics)
// - Tasks sorted by work; OpenMP schedule(static)
// - Fast PRNG backend (default: xoshiro256**), PCG32 available
// - Precomputed Lemire threshold for unbiased bounded ints
// - Packed neighbors (1 load instead of 4)
// - Large batched proposals; manual unroll in hot loop
// - FILE* with big setvbuf; single flush per thread
// NOTE: No fast-math; acceptance is exact double-precision exp.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// -------- Tunables (override with -D... in CMake) --------
#ifndef PROGRESS_MS
#define PROGRESS_MS 2000            // progress print cadence (ms)
#endif
#ifndef BATCH_SIZE
#define BATCH_SIZE 32768            // try 16384..65536; measure on your CPU
#endif
#ifndef BUFFERED_UNIFORMS
#define BUFFERED_UNIFORMS 0         // 0: generate on-the-fly; 1: buffer doubles
#endif
#ifndef RNG_BACKEND
#define RNG_BACKEND 1               // 0: PCG32, 1: xoshiro256**
#endif
// ---------------------------------------------------------

// -------- Progress bar (time-based) --------
static void printProgress(double progress, double elapsed_sec) {
    const int barWidth = 40;
    if (progress < 0) progress = 0;
    if (progress > 1) progress = 1;
    int pos = int(barWidth * progress + 0.5);
    if (pos < 0) pos = 0; if (pos > barWidth) pos = barWidth;

    double pct = progress * 100.0;
    double eta = (progress > 0.0) ? elapsed_sec * (1.0 - progress) / progress : 0.0;
    if (eta < 0) eta = 0;
    int eh = int(eta / 3600), em = int((eta - eh * 3600) / 60), es = int(eta) % 60;

    std::cout << "\r[";
    for (int i = 0; i < barWidth; ++i)
        std::cout << (i < pos ? '=' : (i == pos ? '>' : ' '));
    std::cout << "] " << std::fixed << std::setprecision(1) << pct << "%  ETA "
              << std::setfill('0') << std::setw(2) << eh << ":"
              << std::setw(2) << em << ":" << std::setw(2) << es
              << std::setfill(' ') << std::flush;
}

// -------- Helpers --------
static inline int get_N(double T, double Tc) {
    if (T < 1.0) return 25000;
    else if (T < Tc - 0.5) return 300000;
    else if (T < Tc + 0.5) return 2000000;
    else return 25000;
}
static std::vector<double> make_T_vals(double start, double stop, double step) {
    std::vector<double> vals;
    for (double T = start; T > stop + 1e-12; T += step) vals.push_back(T);
    return vals;
}

// -------- RNG backends --------
#if RNG_BACKEND == 0
// PCG32
struct PCG32 {
    uint64_t state, inc;
    PCG32(uint64_t seed, uint64_t seq=1u) {
        seed ^= 0x853c49e6748fea9bULL;
        inc = (seq<<1u)|1u;
        state = 0; next_u32(); state += seed; next_u32();
    }
    inline uint32_t next_u32() {
        uint64_t old = state;
        state = old * 6364136223846793005ULL + inc;
        uint32_t xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
        uint32_t rot = (uint32_t)(old >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }
    inline double next_double01() { // 53-bit uniform [0,1)
        uint64_t hi = (uint64_t)next_u32() << 21;
        uint64_t lo = next_u32() & ((1u<<21)-1);
        uint64_t r  = hi | lo;
        return (double)r * (1.0/9007199254740992.0);
    }
};
#elif RNG_BACKEND == 1
// xoshiro256** + splitmix64 seeding
static inline uint64_t splitmix64(uint64_t x){
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
struct Xoshiro256ss {
    uint64_t s[4];
    Xoshiro256ss(uint64_t seed, uint64_t seq=1u){
        uint64_t x = seed ^ (seq * 0x9e3779b97f4a7c15ULL);
        s[0] = splitmix64(x);
        s[1] = splitmix64(x + 0x9e3779b97f4a7c15ULL);
        s[2] = splitmix64(x + 2*0x9e3779b97f4a7c15ULL);
        s[3] = splitmix64(x + 3*0x9e3779b97f4a7c15ULL);
        if(!(s[0]|s[1]|s[2]|s[3])) s[0] = 1; // avoid zero state
    }
    static inline uint64_t rotl(uint64_t x, int k){ return (x<<k) | (x>>(64-k)); }
    inline uint64_t next_u64() {
        uint64_t result = rotl(s[1] * 5, 7) * 9;
        uint64_t t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3],45);
        return result;
    }
    inline uint32_t next_u32(){ return (uint32_t)next_u64(); }
    inline double next_double01(){ uint64_t r = next_u64() >> 11; return (double)r * (1.0/9007199254740992.0); }
};
#endif

#if RNG_BACKEND == 0
using RNG = PCG32;
#else
using RNG = Xoshiro256ss;
#endif

// Lemire bounded integer [0,bound), with precomputed threshold t
static inline uint32_t bounded_fast(RNG& rng, uint32_t bound, uint32_t t) {
    uint64_t r = rng.next_u32();
    uint64_t m = r * bound;
    uint32_t l = (uint32_t)m;
    if (l < bound) {
        while (l < t) { r = rng.next_u32(); m = r * bound; l = (uint32_t)m; }
    }
    return (uint32_t)(m >> 32);
}

// -------- Packed neighbors (less pointer chasing) --------
struct Neigh4 { uint16_t u,d,l,r; };

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    constexpr int L = 20, L2 = L*L, J = 1;
    const double T_c = 2.269;
    const int num_per_T = 10;

    const auto T_vals = make_T_vals(10.05, 0.0, -0.05);
    const int nTemps = (int)T_vals.size();

    // Seed base (unique per run)
    std::random_device rd;
    const uint64_t base_seed =
        (uint64_t)rd() ^
        (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();

    // CSV output (FILE* with big userspace buffer)
    FILE* fp = std::fopen("ising_states_cpp.csv", "wb");
    if (!fp) { std::perror("fopen"); return 1; }
    static char io_bigbuf[4<<20]; // 4 MiB
    std::setvbuf(fp, io_bigbuf, _IOFBF, sizeof(io_bigbuf));
    std::mutex io_mutex, print_mutex;

    // ---- Write CSV header ----
    {
        std::string header = "T,phase";
        for (int i = 0; i < L * L; ++i) {
            header += ",spin_" + std::to_string(i);
        }
        header += "\n";
        std::fwrite(header.data(), 1, header.size(), fp);
    }


    // Build tasks and sort by work (large N first) to pair with schedule(static)
    struct Task { int tIdx; int rep; int N; };
    std::vector<Task> tasks; tasks.reserve(nTemps * num_per_T);
    long long total_work = 0;
    for (int t=0; t<nTemps; ++t) {
        int N = get_N(T_vals[t], T_c);
        for (int r=0; r<num_per_T; ++r) {
            tasks.push_back({t,r,N});
            total_work += N;
        }
    }
    std::sort(tasks.begin(), tasks.end(),
              [](const Task& a, const Task& b){ return a.N > b.N; });

    std::atomic<long long> work_done{0};
    const auto t0 = std::chrono::steady_clock::now();

    auto maybe_print = [&](long long w_done) {
        static std::chrono::steady_clock::time_point last = t0;
        auto now = std::chrono::steady_clock::now();
        bool do_print = false;
        { std::lock_guard<std::mutex> lk(print_mutex);
          if (now - last > std::chrono::milliseconds(PROGRESS_MS)) { last = now; do_print = true; } }
        if (do_print) {
            double elapsed = std::chrono::duration<double>(now - t0).count();
            double prog = double(w_done) / double(total_work);
            if (prog < 0.0) prog = 0.0; if (prog > 1.0) prog = 1.0;
            #pragma omp critical
            printProgress(prog, elapsed);
        }
    };

    // Precompute neighbors (packed)
    std::vector<Neigh4> NB(L2);
    for (int i=0;i<L;++i){
        int ip=(i+1==L)?0:i+1, im=(i==0)?L-1:i-1;
        for (int j=0;j<L;++j){
            int jp=(j+1==L)?0:j+1, jm=(j==0)?L-1:j-1;
            int idx=i*L+j;
            NB[idx] = Neigh4{
                (uint16_t)(im*L + j), (uint16_t)(ip*L + j),
                (uint16_t)(i*L + jm), (uint16_t)(i*L + jp)
            };
        }
    }
    const Neigh4* __restrict NBp = NB.data();

    // Lemire constants
    const uint32_t L2U = (uint32_t)L2;
    const uint32_t LEMIRE_T = (uint32_t(-int32_t(L2U)) % L2U);

    // Parallel region
    #pragma omp parallel
    {
        // Per-thread buffers
        std::string outbuf; outbuf.reserve(1<<20); // 1 MiB
        std::vector<int8_t> state(L2);
        std::vector<uint32_t> idx_buf(BATCH_SIZE);
        #if BUFFERED_UNIFORMS
        std::vector<double>   u_buf(BATCH_SIZE);
        #endif

        #pragma omp for schedule(static)
        for (int k=0; k<(int)tasks.size(); ++k) {
            const int tIdx = tasks[k].tIdx;
            const int N    = tasks[k].N;
            const double T = T_vals[tIdx];

            // Unique RNG per task/thread
            uint64_t seed = base_seed ^ (0x9E3779B97F4A7C15ULL*(tIdx+1) + (uint64_t)k);
            #ifdef _OPENMP
            seed ^= (uint64_t)omp_get_thread_num();
            #endif
            RNG rng(seed, seed + 0x5851f42d4c957f2dULL);

            // Exact Metropolis acceptance lookup (dE ∈ {-8,-4,0,4,8})
            double p_lut[17] = {0};
            p_lut[8+4] = std::exp(-4.0 / T);
            p_lut[8+8] = std::exp(-8.0 / T);

            // Fast spin init: 32 spins from one 32-bit word
            for (int i=0; i<L2; i+=32) {
                uint32_t r = (uint32_t)rng.next_u32();
                int lim = std::min(32, L2 - i);
                for (int b=0; b<lim; ++b) state[i+b] = ((r>>b)&1u) ? 1 : -1;
            }

            int8_t* __restrict sptr = state.data();

            auto do_batch = [&](int count){
                for (int b=0; b<count; ++b) idx_buf[b] = bounded_fast(rng, L2U, LEMIRE_T);
                #if BUFFERED_UNIFORMS
                for (int b=0; b<count; ++b) u_buf[b] = rng.next_double01();
                #endif

                int b=0;
                for (; b+3 < count; b+=4) {
                    const int i0 = (int)idx_buf[b+0];
                    const int i1 = (int)idx_buf[b+1];
                    const int i2 = (int)idx_buf[b+2];
                    const int i3 = (int)idx_buf[b+3];

                    const Neigh4 n0 = NBp[i0], n1 = NBp[i1], n2 = NBp[i2], n3 = NBp[i3];

                    const int s0 = sptr[i0], s1 = sptr[i1], s2 = sptr[i2], s3 = sptr[i3];

                    const int nb0 = sptr[n0.u] + sptr[n0.d] + sptr[n0.l] + sptr[n0.r];
                    const int nb1 = sptr[n1.u] + sptr[n1.d] + sptr[n1.l] + sptr[n1.r];
                    const int nb2 = sptr[n2.u] + sptr[n2.d] + sptr[n2.l] + sptr[n2.r];
                    const int nb3 = sptr[n3.u] + sptr[n3.d] + sptr[n3.l] + sptr[n3.r];

                    const int dE0 = 2*J*s0*nb0;
                    const int dE1 = 2*J*s1*nb1;
                    const int dE2 = 2*J*s2*nb2;
                    const int dE3 = 2*J*s3*nb3;

                    const double a0 = (dE0<=0)?1.0:p_lut[dE0+8];
                    const double a1 = (dE1<=0)?1.0:p_lut[dE1+8];
                    const double a2 = (dE2<=0)?1.0:p_lut[dE2+8];
                    const double a3 = (dE3<=0)?1.0:p_lut[dE3+8];

                    #if BUFFERED_UNIFORMS
                        if (u_buf[b+0] < a0) sptr[i0] = (int8_t)(-s0);
                        if (u_buf[b+1] < a1) sptr[i1] = (int8_t)(-s1);
                        if (u_buf[b+2] < a2) sptr[i2] = (int8_t)(-s2);
                        if (u_buf[b+3] < a3) sptr[i3] = (int8_t)(-s3);
                    #else
                        if (rng.next_double01() < a0) sptr[i0] = (int8_t)(-s0);
                        if (rng.next_double01() < a1) sptr[i1] = (int8_t)(-s1);
                        if (rng.next_double01() < a2) sptr[i2] = (int8_t)(-s2);
                        if (rng.next_double01() < a3) sptr[i3] = (int8_t)(-s3);
                    #endif
                }
                for (; b < count; ++b) {
                    const int idx = (int)idx_buf[b];
                    const Neigh4 n4 = NBp[idx];
                    const int s = sptr[idx];
                    const int nb = sptr[n4.u] + sptr[n4.d] + sptr[n4.l] + sptr[n4.r];
                    const int dE = 2*J*s*nb;
                    const double a = (dE<=0)?1.0:p_lut[dE+8];
                    #if BUFFERED_UNIFORMS
                        if (u_buf[b] < a) sptr[idx] = (int8_t)(-s);
                    #else
                        if (rng.next_double01() < a) sptr[idx] = (int8_t)(-s);
                    #endif
                }
            };

            // N sweeps; each sweep = L2 proposals processed in batches
            for (int sweep=0; sweep<N; ++sweep) {
                int rem = L2;
                while (rem > 0) {
                    int take = (rem < BATCH_SIZE) ? rem : BATCH_SIZE;
                    do_batch(take);
                    rem -= take;
                }
            }

            // End-of-task progress (exactly N added)
            { long long w = work_done.fetch_add(N) + N; maybe_print(w); }


            // Compose one CSV row into per-thread buffer
            outbuf.append(std::to_string(T));
            outbuf.push_back(',');
            outbuf.push_back((T < T_c) ? '1' : '0');
            for (int q=0; q<L2; ++q) {
                outbuf.push_back(',');
                if (sptr[q] > 0) outbuf.push_back('1');
                else { outbuf.push_back('-'); outbuf.push_back('1'); }
            }
            outbuf.push_back('\n');
        } // tasks

        // Single flush per thread
        if (!outbuf.empty()) {
            std::lock_guard<std::mutex> lk(io_mutex);
            std::fwrite(outbuf.data(), 1, outbuf.size(), fp);
        }
    } // omp parallel

    std::fclose(fp);
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printProgress(1.0, elapsed);
    std::cout << "\nSaved to ising_states_cpp.csv  (elapsed "
              << std::fixed << std::setprecision(1) << elapsed << " s)\n";
    return 0;
}
