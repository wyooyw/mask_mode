#include <torch/extension.h>

#include <iostream>
#include <vector>
#include "mask_mode_cuda.cu"

at::Tensor mask_mode_cuda(torch::Tensor tensor_a, torch::Tensor tensor_mask, int grid_size, int block_size);

#define MAX_BLOCK_SIZE 1024
#define MAX_GRID_SIZE 65536

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DIM(x) TORCH_CHECK(x.dim()==2, #x " must has only 2 dims")
#define CHECK_DTYPE(x) TORCH_CHECK(x.dtype()==torch::kInt8, #x " must be torch.int8")
#define CHECK_SIZE(x) TORCH_CHECK(x.size(0)<=MAX_GRID_SIZE, #x " 's row must <= 65536")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_DIM(x); CHECK_DTYPE(x); CHECK_SIZE(x)

#define CHECK_RANGE(x) TORCH_CHECK(x.le(8).all().item<bool>() && x.ge(0).all().item<bool>(), #x " must in range [0,8]")

at::Tensor mask_mode(
    torch::Tensor tensor_in, torch::Tensor tensor_mask) {
    CHECK_INPUT(tensor_in);
    CHECK_INPUT(tensor_mask);
    CHECK_RANGE(tensor_in);
    
    int grid_size = tensor_in.size(0);
    int block_size = tensor_in.size(1) < MAX_BLOCK_SIZE ? tensor_in.size(1) : MAX_BLOCK_SIZE;

    auto result = mask_mode_cuda(tensor_in, tensor_mask, grid_size, block_size);
    return result;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mask_mode", &mask_mode, "custom mode forward");
//   m.def("backward", &toy_matmul_backward, "Toy mamtul backward");
}