#pragma once

#include <cstddef>
#include <cuda_runtime.h>

template <typename T, size_t N> struct Array {
    T arr[N];

    __device__ __host__ Array(T *raw_ptr) {
        for (size_t i = 0; i < N; i++) {
            arr[i] = raw_ptr[i];
        }
    }

    __device__ __host__ T operator[](size_t i) { return arr[i]; }
    __device__ __host__ const T operator[](size_t i) const { return arr[i]; }
};

template <typename T, size_t N> struct PtrArray {
    T *arr[N];

    __device__ __host__ PtrArray(T **raw_ptr) {
        for (size_t i = 0; i < N; i++) {
            arr[i] = raw_ptr[i];
        }
    }

    __device__ __host__ T *operator[](size_t i) { return arr[i]; }
    __device__ __host__ const T *operator[](size_t i) const { return arr[i]; }
};

/// Small host-metadata arrays used by recursion trace generation used to be
/// passed to kernels by value.  That limits the verifier to a handful of
/// proofs because CUDA's kernel-parameter area is finite.  This RAII helper
/// stages such arrays in device memory instead.  The upload is synchronized
/// explicitly by the caller before the host vector may be released; the
/// asynchronous free remains ordered after all kernels on the same stream.
template <typename T> class DeviceArrayCopy {
  private:
    T *ptr_ = nullptr;
    cudaStream_t stream_ = nullptr;
    cudaError_t status_ = cudaSuccess;

  public:
    DeviceArrayCopy(const T *host, size_t count, cudaStream_t stream) : stream_(stream) {
        if (count == 0) {
            return;
        }
        status_ = cudaMallocAsync(reinterpret_cast<void **>(&ptr_), count * sizeof(T), stream_);
        if (status_ != cudaSuccess) {
            ptr_ = nullptr;
            return;
        }
        status_ = cudaMemcpyAsync(
            ptr_, host, count * sizeof(T), cudaMemcpyHostToDevice, stream_
        );
    }

    DeviceArrayCopy(const DeviceArrayCopy &) = delete;
    DeviceArrayCopy &operator=(const DeviceArrayCopy &) = delete;

    ~DeviceArrayCopy() {
        if (ptr_ != nullptr) {
            cudaFreeAsync(ptr_, stream_);
        }
    }

    cudaError_t status() const { return status_; }
    T *get() { return ptr_; }
    const T *get() const { return ptr_; }
};
