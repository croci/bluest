#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
//#include <pybind11/stl.h>
//#include <pybind11/pytypes.h>
//#include <pybind11/cast.h>
//#include <pybind11/operators.h>

namespace py = pybind11;

void assemble_psi_c(py::array_t<double, py::array::c_style> psi, const int N, const int k, const int Lk, const py::array_t<long int, py::array::c_style> groupsk, const py::array_t<double, py::array::c_style> invcovsk){
    const int ksq = k*k;
    double * psi_x = (double*) psi.data(0);
    const long int * groupsk_x = groupsk.data(0);
    const double * invcovsk_x = invcovsk.data(0);
    for(int i=0; i<Lk; i++){
        for(int j=0; j<k; j++){
            for(int l=0; l<k; l++){
                psi_x[Lk*(N*groupsk_x[k*i+j]+groupsk_x[k*i+l]) + i] += invcovsk_x[ksq*i + k*j + l];
            }
        }
    }
    
}

template <typename T>
void objectiveK_c(py::array_t<double, py::array::c_style> PHI, const int N, const int k, const int Lk, const py::array_t<T, py::array::c_style> mk, const py::array_t<long int, py::array::c_style> groupsk, const py::array_t<double, py::array::c_style> invcovsk){
    const int ksq = k*k;
    double * PHI_x = (double*) PHI.data(0);
    const T * mk_x = mk.data(0);
    const long int * groupsk_x = groupsk.data(0);
    const double * invcovsk_x = invcovsk.data(0);
    for(int i=0; i<Lk; i++){
        for(int j=0; j<k; j++){
            for(int l=0; l<k; l++){
                PHI_x[N*groupsk_x[k*i+j]+groupsk_x[k*i+l]] += mk_x[i]*invcovsk_x[ksq*i + k*j + l];
            }
        }
    }
    
}

void cleanupK_c(py::array_t<double, py::array::c_style> X, const int k, const int Lk, const py::array_t<long int, py::array::c_style> groupsk, const py::array_t<double, py::array::c_style> invcovsk, const py::array_t<double, py::array::c_style> invPHI_0){
    const int ksq = k*k;
    double * X_x = (double*) X.data(0);
    const long int * groupsk_x = groupsk.data(0);
    const double * invcovsk_x  = invcovsk.data(0);
    const double * invPHI_0_x  = invPHI_0.data(0);
    for(int i=0; i<Lk; i++){
        for(int j=0; j<k; j++){
            for(int l=0; l<k; l++){
                X_x[Lk*groupsk_x[k*i+j] + i] = invcovsk_x[ksq*i + k*j + l]*invPHI_0_x[groupsk_x[k*i+l]];
            }
        }
    }

}

void gradK_c(py::array_t<double, py::array::c_style> grad, const int k, const int Lk, const py::array_t<long int, py::array::c_style> groupsk, const py::array_t<double, py::array::c_style> invcovsk, const py::array_t<double, py::array::c_style> invPHI_0){
    const int ksq = k*k;
    double * grad_x = (double*) grad.data(0);
    const long int * groupsk_x = groupsk.data(0);
    const double * invcovsk_x = invcovsk.data(0);
    const double * invPHI_0_x = invPHI_0.data(0);
    for(int i=0; i<Lk; i++){
        for(int j=0; j<k; j++){
            for(int l=0; l<k; l++){
                grad_x[i] += invPHI_0_x[groupsk_x[k*i+j]]*invcovsk_x[ksq*i + k*j + l]*invPHI_0_x[groupsk_x[k*i+l]];
            }
        }
    }
    
}

void hessKQ_c(py::array_t<double, py::array::c_style> hess, const int N, const int k, const int q, const int Lk, const int Lq, const py::array_t<long int, py::array::c_style> groupsk, const py::array_t<long int, py::array::c_style> groupsq, const py::array_t<double, py::array::c_style> invcovsk, const py::array_t<double, py::array::c_style> invcovsq, const py::array_t<double, py::array::c_style> invPHI){
    const int ksq = k*k;
    const int qsq = q*q;
    double * hess_x = (double*) hess.data(0);
    const long int * groupsk_x = groupsk.data(0);
    const long int * groupsq_x = groupsq.data(0);
    const double * invcovsk_x = invcovsk.data(0);
    const double * invcovsq_x = invcovsq.data(0);
    const double * invPHI_x = invPHI.data(0);
    for(int ik=0; ik<Lk; ik++){
        for(int iq=0; iq<Lq; iq++){
            for(int lk=0; lk<k; lk++){
                for(int jk=0; jk<k; jk++){
                    for(int jq=0; jq<q; jq++){
                        for(int lq=0; lq<q; lq++){
                            hess_x[ik*Lq + iq] += invPHI_x[groupsk_x[k*ik+lk]]*invcovsk_x[ksq*ik + k*lk + jk]*invPHI_x[N*groupsk_x[k*ik+jk] + groupsq_x[q*iq+jq]]*invcovsq_x[qsq*iq + q*jq + lq]*invPHI_x[groupsq_x[q*iq+lq]];
                        }
                    }
                }
            }
        }
    }

}

PYBIND11_MODULE(cmisc, m) {
    m.doc() = "pybind11 wrapper for cmisc";

    m.def("assemble_psi_c", &assemble_psi_c);
    m.def("objectiveK_c", &objectiveK_c<double>);
    m.def("objectiveK_c", &objectiveK_c<long int>);
    m.def("gradK_c", &gradK_c);
    m.def("hessKQ_c", &hessKQ_c);
    m.def("cleanupK_c", &cleanupK_c);

}
