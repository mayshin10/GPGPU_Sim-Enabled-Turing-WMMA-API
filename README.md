# Abstract
  This is a repository for my undergraduate thesis, *_Modeling Tensor Core Microarchitecture for NVIDIA Turing Architecture with Experimental Features_*.
  In this study, the microarchitecture of Tensor Core in Turing architecture is proposed. Since NVIDIA does not disclose the inside of the tensor core, it is necessary to profile through microbenchmarking. Dissecting the NVIDIA GPUs has also been done in previous studies. However, it was not revealed about the experimental features of the Turing architecture, i.e. INT4(int 4-bit) operation mode and B1(binary 1-bit) operation mode. All of these functions were analyzed in this study.

# File Structure
* Benchmark<br>
each directory represented each data type su
   * b1
   * u4
   * u8
   * fp16
   * mixed

* GPGPU-Sim



The current(2020.12) GPGPU-Sim supports up to the 1st Gen NVIDIA tensor core. 

This distribution includes the simulator that supports up to the 2nd Gen tensor core(Turing arch).

It is also my Undergraduate graduation thesis, Yonsei Univ, Korea. 
