#include <iostream>

#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>

#include <cutlass/util/host_tensor.h>

int main()
{

    // Gemm operation
    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t, // A type + layout
        cutlass::layout::ColumnMajor,
        cutlass::half_t, // B
        cutlass::layout::ColumnMajor,
        cutlass::half_t, // C
        cutlass::layout::ColumnMajor,
        float,                          // accumulator type
        cutlass::arch::OpClassTensorOp, // what operation to perform by the GEMM (tensor ops)
        cutlass::arch::Sm80>;          // architecture to target(let's target ampere for now)

    Gemm gemm_op;
    cutlass::Status status;

    int M = 512;
    int N = 512;
    int K = 512;

    float alpha = 1.25f;
    float beta = 1.25f;

    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> A({M, K});
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> B({K, N});
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> C({M, N});

    // HostTensor will allocate space on GPU. Here, we can access the pointers
    cutlass::half_t const *ptrA = A.device_data();
    cutlass::half_t const *ptrB = B.device_data();
    cutlass::half_t const *ptrC = C.device_data();
    cutlass::half_t *ptrD = C.device_data();

    // Device_Ref is a TensorRef that points to tensor in device memory, and also contains other data. We get stride(0) for lead dim bcz eg. MxK, stride(0) is K
    int lda = A.device_ref().stride(0);
    int ldb = B.device_ref().stride(0);
    int ldc = C.device_ref().stride(0);
    int ldd = C.device_ref().stride(0);

    cutlass::gemm::GemmCoord problem_size(M, N, K);

    // Launch GEMM
    status = gemm_op({
        {M, N, K},
        {ptrA, lda},  // TensorRef to A device tensor
        {ptrB, ldb},  // TensorRef to B device tensor
        {ptrC, ldc},  // TensorRef to C device tensor
        {ptrD, ldd},  // TensorRef to D device tensor - may be the same as C
        {alpha, beta} // epilogue operation arguments
    });

    if (status != cutlass::Status::kSuccess) {
        std::cout << "helppp " << cutlass::cutlassGetStatusString(status) << std::endl;
        return -1;
    }
    return 0;
}