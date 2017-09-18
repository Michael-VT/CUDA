#include <iostream>

namespace JL{
    template <class T>
    inline void print_array1d(T a, int n){
        for(int i=0;i<n;++i){
            std::cout<<a[i]<<' ';
        }
        std::cout<<'\n';
    }

    template <class T>
    inline void print_array2d(T a, int n, int m){
        for(int i=0;i<n;++i){
            for(int j=0; j<m; ++j){
                std::cout<<a[i][j]<<' ';
            }
            std::cout<<'\n';
        }
    }

}

