#ifndef PYBIND11_DIR
#	PYBIND11_DIR := /usr
#endif

INCL := -I${PYBIND11_DIR}/include # -I/usr/include/x86_64-linux-gnu/c++/9

CXX	:= g++
FLAGS	:= -std=c++11 -O3 -m64 -ftree-vectorize -fopt-info-vec-optimized -ffast-math -march=native -Wl,--no-as-needed 
FLAGSD	:= -std=c++11 -g -O0 -Wl,--no-as-needed 

default:
	$(CXX) $(FLAGS) $(INCL) -shared -fPIC `python3 -m pybind11 --includes` ./bluest/cmisc.cpp -o cmisc`python3-config --extension-suffix` $(LIB)
	mv *.so ./bluest/

debug:
	$(CXX) $(FLAGSD) $(INCL) -shared -fPIC `python3 -m pybind11 --includes` ./bluest/cmisc.cpp -o cmisc`python3-config --extension-suffix` $(LIB)
	mv *.so ./bluest/

clean:
	rm -f ./bluest/*.so
