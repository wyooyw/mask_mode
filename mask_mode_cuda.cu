#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define CEIL(x,n) ((x+n-1)/n)

template <typename scalar_t>
__global__ void mask_mode_cuda_kernel(
    scalar_t* tensor_in, // row * column
    scalar_t* tensor_mask, // row * column
    int* tensor_out, // row
    int row, // tensor_in的行数
    int column, // tensor_in的列数
    int ele_num_per_thread // 每个线程处理的元素数量
    ) {
      __shared__ int counts[1024 * 9];
      int block_idx = blockIdx.x;
      int thread_idx = threadIdx.x;

      int input_row_idx = block_idx; // 当前block处理的行
      int input_begin_addr = input_row_idx * column;
      int input_col_idx_begin = thread_idx * ele_num_per_thread; // 当前thread处理的起始列
      int input_col_idx_end = min(input_col_idx_begin + ele_num_per_thread, column); // 当前thread处理的终止列(不包含)

      int counts_begin_addr = thread_idx * 9;
      int output_begin_addr = block_idx;

      int activate_thread_num = CEIL(column,ele_num_per_thread);
      int mode_val = -1;
      int mode_cnt = -1;

      if(input_row_idx < row){
        if(input_col_idx_begin < column){

          // 清零
          for(int i = 0; i < 9;i++){
            counts[counts_begin_addr + i] = 0;
          }

          // 线程内计数
          for(int i = input_col_idx_begin; i<input_col_idx_end;i++){
            scalar_t ele = tensor_in[input_begin_addr + i];
            counts[counts_begin_addr + ele] += (1 & tensor_mask[input_begin_addr + i]);
          }
          
        }

        // 线程间累加
        int acc_thread_num = activate_thread_num / 2;
        int acc_thread_num_ceil = CEIL(activate_thread_num, 2);
        do{
          
          if(thread_idx < acc_thread_num){
            for(int i = 0 ; i < 9; i++){
              counts[thread_idx * 9 + i] += counts[(thread_idx + acc_thread_num_ceil) * 9 + i];
            }
          }
           __syncthreads();
          acc_thread_num = acc_thread_num_ceil / 2 ;
          acc_thread_num_ceil = CEIL(acc_thread_num_ceil, 2);
        }while(acc_thread_num > 0);
        
        __syncthreads();

        // 将counts拷贝到tensor_out
        if(thread_idx==0){
          for(int i = 0 ; i < 9 ; i++){
            if(counts[i] > mode_cnt){
              mode_cnt = counts[i];
              mode_val = i;
            }
          }
          tensor_out[output_begin_addr] = mode_val;
        }
      }

}

torch::Tensor mask_mode_cuda(
    torch::Tensor tensor_in, torch::Tensor tensor_mask,int grid_size, int block_size) {
  int row = tensor_in.size(0);
  int column = tensor_in.size(1);

  int ele_num_per_thread = CEIL(column, block_size);;
  

  auto tensor_out = torch::zeros({row},
                     torch::dtype(torch::kInt).device(tensor_in.device()));

  AT_DISPATCH_INTEGRAL_TYPES(tensor_in.type(), "mask_mode_cuda", ([&] {
    mask_mode_cuda_kernel<scalar_t><<<grid_size, block_size>>>(
        tensor_in.data<scalar_t>(),
        tensor_mask.data<scalar_t>(),
        tensor_out.data<int>(),
        row,
        column,
        ele_num_per_thread
        );
  }));
 
  return tensor_out;
}


template <typename scalar_t>
__global__ void mode_cuda_kernel(
    scalar_t* tensor_in, // row * column
    int* tensor_out, // row
    int row, // tensor_in的行数
    int column, // tensor_in的列数
    int ele_num_per_thread // 每个线程处理的元素数量
    ) {
      __shared__ int counts[1024 * 9];
      int block_idx = blockIdx.x;
      int thread_idx = threadIdx.x;

      int input_row_idx = block_idx; // 当前block处理的行
      int input_begin_addr = input_row_idx * column;
      int input_col_idx_begin = thread_idx * ele_num_per_thread; // 当前thread处理的起始列
      int input_col_idx_end = min(input_col_idx_begin + ele_num_per_thread, column); // 当前thread处理的终止列(不包含)

      int counts_begin_addr = thread_idx * 9;
      int output_begin_addr = block_idx;

      int activate_thread_num = CEIL(column,ele_num_per_thread);
      int mode_val = -1;
      int mode_cnt = -1;

      if(input_row_idx < row){
        if(input_col_idx_begin < column){

          // 清零
          for(int i = 0; i < 9;i++){
            counts[counts_begin_addr + i] = 0;
          }

          // 线程内计数
          for(int i = input_col_idx_begin; i<input_col_idx_end;i++){
            scalar_t ele = tensor_in[input_begin_addr + i];
            counts[counts_begin_addr + ele] += 1;
          }
          
        }

        // 线程间累加
        int acc_thread_num = activate_thread_num / 2;
        int acc_thread_num_ceil = CEIL(activate_thread_num, 2);
        do{
          
          if(thread_idx < acc_thread_num){
            for(int i = 0 ; i < 9; i++){
              counts[thread_idx * 9 + i] += counts[(thread_idx + acc_thread_num_ceil) * 9 + i];
            }
          }
           __syncthreads();
          acc_thread_num = acc_thread_num_ceil / 2 ;
          acc_thread_num_ceil = CEIL(acc_thread_num_ceil, 2);
        }while(acc_thread_num > 0);
        
        __syncthreads();

        // 将counts拷贝到tensor_out
        if(thread_idx==0){
          for(int i = 0 ; i < 9 ; i++){
            if(counts[i] > mode_cnt){
              mode_cnt = counts[i];
              mode_val = i;
            }
          }
          tensor_out[output_begin_addr] = mode_val;
        }
      }

}

torch::Tensor mode_cuda(
    torch::Tensor tensor_in, int grid_size, int block_size) {
  int row = tensor_in.size(0);
  int column = tensor_in.size(1);

  int ele_num_per_thread = CEIL(column, block_size);;
  

  auto tensor_out = torch::zeros({row},
                     torch::dtype(torch::kInt).device(tensor_in.device()));

  AT_DISPATCH_INTEGRAL_TYPES(tensor_in.type(), "mode_cuda", ([&] {
    mode_cuda_kernel<scalar_t><<<grid_size, block_size>>>(
        tensor_in.data<scalar_t>(),
        tensor_out.data<int>(),
        row,
        column,
        ele_num_per_thread
        );
  }));
 
  return tensor_out;
}