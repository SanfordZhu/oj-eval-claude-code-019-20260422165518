#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  size_t d = keys[0]->GetColumnNum();

  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    size_t r = i + 1;

    // Build K [r, d] in SRAM
    gpu_sim.MoveMatrixToSharedMem(keys[0]);
    gpu_sim.Run(false, &matrix_memory_allocator);
    Matrix *K = matrix_memory_allocator.Allocate("K");
    gpu_sim.Copy(keys[0], K, kInSharedMemory);
    gpu_sim.Run(false, &matrix_memory_allocator);
    gpu_sim.MoveMatrixToGpuHbm(keys[0]);
    gpu_sim.Run(false, &matrix_memory_allocator);

    for (size_t j = 1; j <= i; ++j) {
      gpu_sim.MoveMatrixToSharedMem(keys[j]);
      gpu_sim.Run(false, &matrix_memory_allocator);
      Matrix *nk = matrix_memory_allocator.Allocate("K2");
      gpu_sim.Concat(K, keys[j], nk, 0, kInSharedMemory);
      gpu_sim.Run(false, &matrix_memory_allocator);
      gpu_sim.ReleaseMatrix(K);
      gpu_sim.MoveMatrixToGpuHbm(keys[j]);
      gpu_sim.Run(false, &matrix_memory_allocator);
      K = nk;
    }

    // Move Q to SRAM
    gpu_sim.MoveMatrixToSharedMem(current_query);
    gpu_sim.Run(false, &matrix_memory_allocator);

    // Transpose K: [r, d] -> [d, r]
    gpu_sim.Transpose(K, kInSharedMemory);
    gpu_sim.Run(false, &matrix_memory_allocator);

    // S = Q @ K^T: [r, d] @ [d, r] = [r, r]
    Matrix *S = matrix_memory_allocator.Allocate("S");
    gpu_sim.MatMul(current_query, K, S);
    gpu_sim.Run(false, &matrix_memory_allocator);

    // Release K and Q
    gpu_sim.ReleaseMatrix(K);
    gpu_sim.ReleaseMatrix(current_query);
    gpu_sim.Run(false, &matrix_memory_allocator);

    // exp_S = exp(S): [r, r]
    Matrix *exp_S = matrix_memory_allocator.Allocate("expS");
    gpu_sim.MatExp(S, exp_S);
    gpu_sim.Run(false, &matrix_memory_allocator);
    gpu_sim.ReleaseMatrix(S);
    gpu_sim.Run(false, &matrix_memory_allocator);

    // Build V [r, d] in SRAM
    gpu_sim.MoveMatrixToSharedMem(values[0]);
    gpu_sim.Run(false, &matrix_memory_allocator);
    Matrix *V = matrix_memory_allocator.Allocate("V");
    gpu_sim.Copy(values[0], V, kInSharedMemory);
    gpu_sim.Run(false, &matrix_memory_allocator);
    gpu_sim.MoveMatrixToGpuHbm(values[0]);
    gpu_sim.Run(false, &matrix_memory_allocator);

    for (size_t j = 1; j <= i; ++j) {
      gpu_sim.MoveMatrixToSharedMem(values[j]);
      gpu_sim.Run(false, &matrix_memory_allocator);
      Matrix *nv = matrix_memory_allocator.Allocate("V2");
      gpu_sim.Concat(V, values[j], nv, 0, kInSharedMemory);
      gpu_sim.Run(false, &matrix_memory_allocator);
      gpu_sim.ReleaseMatrix(V);
      gpu_sim.MoveMatrixToGpuHbm(values[j]);
      gpu_sim.Run(false, &matrix_memory_allocator);
      V = nv;
    }

    // Build softmax matrix P [r, r] in HBM row by row
    // For each row: P[row] = exp_S[row] / sum(exp_S[row])

    // Process row 0
    Matrix *row0 = matrix_memory_allocator.Allocate("r0");
    gpu_sim.GetRow(exp_S, 0, row0, kInSharedMemory);
    gpu_sim.Run(false, &matrix_memory_allocator);

    Matrix *sum0 = matrix_memory_allocator.Allocate("s0");
    gpu_sim.Sum(row0, sum0);
    gpu_sim.Run(false, &matrix_memory_allocator);

    Matrix *sm_row0 = matrix_memory_allocator.Allocate("sm0");
    gpu_sim.MatDiv(row0, sum0, sm_row0);
    gpu_sim.Run(false, &matrix_memory_allocator);
    gpu_sim.ReleaseMatrix(row0);
    gpu_sim.ReleaseMatrix(sum0);

    gpu_sim.MoveMatrixToGpuHbm(sm_row0);
    gpu_sim.Run(false, &matrix_memory_allocator);

    Matrix *P = matrix_memory_allocator.Allocate("P");
    gpu_sim.Copy(sm_row0, P, kInGpuHbm);
    gpu_sim.Run(false, &matrix_memory_allocator);
    gpu_sim.ReleaseMatrix(sm_row0);

    for (size_t ri = 1; ri < r; ++ri) {
      Matrix *re = matrix_memory_allocator.Allocate("re");
      gpu_sim.GetRow(exp_S, ri, re, kInSharedMemory);
      gpu_sim.Run(false, &matrix_memory_allocator);

      Matrix *rsum = matrix_memory_allocator.Allocate("rsum");
      gpu_sim.Sum(re, rsum);
      gpu_sim.Run(false, &matrix_memory_allocator);

      Matrix *sm_ri = matrix_memory_allocator.Allocate("sm");
      gpu_sim.MatDiv(re, rsum, sm_ri);
      gpu_sim.Run(false, &matrix_memory_allocator);
      gpu_sim.ReleaseMatrix(re);
      gpu_sim.ReleaseMatrix(rsum);

      gpu_sim.MoveMatrixToGpuHbm(sm_ri);
      gpu_sim.Run(false, &matrix_memory_allocator);

      Matrix *new_P = matrix_memory_allocator.Allocate("P2");
      gpu_sim.Concat(P, sm_ri, new_P, 0, kInGpuHbm);
      gpu_sim.Run(false, &matrix_memory_allocator);
      gpu_sim.ReleaseMatrix(P);
      gpu_sim.ReleaseMatrix(sm_ri);
      P = new_P;
    }

    gpu_sim.ReleaseMatrix(exp_S);

    // Move P to SRAM for MatMul with V
    gpu_sim.MoveMatrixToSharedMem(P);
    gpu_sim.Run(false, &matrix_memory_allocator);

    // answer = P @ V: [r, r] @ [r, d] = [r, d]
    Matrix *answer = matrix_memory_allocator.Allocate("ans");
    gpu_sim.MatMul(P, V, answer);
    gpu_sim.Run(false, &matrix_memory_allocator);

    gpu_sim.ReleaseMatrix(P);
    gpu_sim.ReleaseMatrix(V);
    gpu_sim.Run(false, &matrix_memory_allocator);

    // Move answer to HBM
    gpu_sim.MoveMatrixToGpuHbm(answer);
    gpu_sim.Run(false, &matrix_memory_allocator);

    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*answer);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
