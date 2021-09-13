#include <iostream>
#include <fstream>
#include <ctime>

#include <CL/sycl.hpp>

using namespace cl::sycl;

const int N = 1024*1024; //default vector size

int test_shared_memory(size_t length){

    clock_t start = clock();

    sycl::queue q(sycl::default_selector{});

    auto X = sycl::malloc_shared<float>(length, q);
    auto Y = sycl::malloc_shared<float>(length, q);
    auto Z = sycl::malloc_shared<float>(length, q);
    auto W = sycl::malloc_shared<float>(length, q);

    for(int i = 0; i < length; i ++){
        X[i] = (float)rand() / (RAND_MAX + 1.0);
        Y[i] = (float)rand() / (RAND_MAX + 1.0);
        Z[i] = (float)rand() / (RAND_MAX + 1.0);
        W[i] = 0.0;
    }

    try {

        q.parallel_for( sycl::range<1>{length}, [=] (sycl::id<1> i) {
            W[i] = X[i] * Y[i] + Z[i];
        });
        q.wait();
    }
    catch (sycl::exception & e) {
        std::cout << e.what() << std::endl;
        return 1;
    }

    std::cout << "shared memory time consuming：" << (clock() - start) / 1000 << "ms" << std::endl;

    size_t errors(0);
    for(int i = 0; i < length; i ++){
        float val = X[i] * Y[i] + Z[i];
        if( std::abs(W[i] - val) > 1E-4){
            std::cout << i << " error " << W[i] << "\t" << val << std::endl;
            errors ++;
        }           
    }

    sycl::free(X, q);
    sycl::free(Y, q);
    sycl::free(Z, q);
    sycl::free(W, q);

    return 0;    
}


int test_host_memory(size_t length){

    clock_t start = clock();

    sycl::queue q(sycl::default_selector{});

    auto X = sycl::malloc_host<float>(length, q);
    auto Y = sycl::malloc_host<float>(length, q);
    auto Z = sycl::malloc_host<float>(length, q);
    auto W = sycl::malloc_host<float>(length, q);

    for(int i = 0; i < length; i ++){
        X[i] = (float)rand() / (RAND_MAX + 1.0);
        Y[i] = (float)rand() / (RAND_MAX + 1.0);
        Z[i] = (float)rand() / (RAND_MAX + 1.0);
        W[i] = 0.0;
    }

    try {

        q.parallel_for( sycl::range<1>{length}, [=] (sycl::id<1> i) {
            W[i] = X[i] * Y[i] + Z[i];
        });
        q.wait();
    }
    catch (sycl::exception & e) {
        std::cout << e.what() << std::endl;
        return 1;
    }

    std::cout << "host memory time consuming：" << (clock() - start) / 1000 << "ms" << std::endl;

    size_t errors(0);
    for(int i = 0; i < length; i ++){
        float val = X[i] * Y[i] + Z[i];
        if( std::abs(W[i] - val) > 1E-4){
            std::cout << i << " error " << W[i] << "\t" << val << std::endl;
            errors ++;
        }           
    }

    sycl::free(X, q);
    sycl::free(Y, q);
    sycl::free(Z, q);
    sycl::free(W, q);

    return 0;    
}

int test_device_memory(size_t length){

    clock_t start = clock();
    std::vector<float> h_X(length);
    std::vector<float> h_Y(length);
    std::vector<float> h_Z(length);
    std::vector<float> h_W(length, 0.0);

    for( int i = 0; i < length; i++){
        h_X[i] =  (float)rand() / (RAND_MAX + 1.0);
        h_Y[i] =  (float)rand() / (RAND_MAX + 1.0);
        h_Z[i] =  (float)rand() / (RAND_MAX + 1.0);
    }

    
    sycl::queue q(sycl::gpu_selector{});

    auto X = sycl::malloc_device<float>(length, q);
    auto Y = sycl::malloc_device<float>(length, q);
    auto Z = sycl::malloc_device<float>(length, q);
    auto W = sycl::malloc_device<float>(length, q);

    const size_t bytes = length * sizeof(float);

    q.memcpy(X, h_X.data(), bytes);
    q.memcpy(Y, h_Y.data(), bytes);
    q.memcpy(Z, h_Z.data(), bytes);
    q.wait();

    try {

        q.parallel_for( sycl::range<1>{length}, [=] (sycl::id<1> i) {
            W[i] = X[i] * Y[i] + Z[i];
        });
        q.wait();
    }
    catch (sycl::exception & e) {
        std::cout << e.what() << std::endl;
        return 1;
    }

    q.memcpy(h_W.data(), W, bytes);
    q.wait();
    std::cout << "device memory time consuming：" << (clock() - start) / 1000 << "ms" << std::endl;

    size_t errors(0);
    for(int i = 0; i < length; i ++){
        float val = h_X[i] * h_Y[i] + h_Z[i];
        if( std::abs(h_W[i] - val) > 1E-4){
            std::cout << i << " error " << h_W[i] << "\t" << val << std::endl;
            errors ++;
        }        
    }

    sycl::free(X, q);
    sycl::free(Y, q);
    sycl::free(Z, q);
    sycl::free(W, q);

    return 0;
}

int main(int argc, char **argv){
    size_t length = 0;
    if(argc < 2)
        length = N;
    else
        length = std::atoi(argv[1]);

    std::cout << "Runing with vector size: " << length << std::endl;

    std::cout << "shared memory test..." << std::endl;
    test_shared_memory(length);

    std::cout << "host memory test..." << std::endl;
    test_host_memory(length);

    std::cout << "device memory test..." << std::endl;
    test_device_memory(length);   

    return 0;

}