#ifndef RK_HPP
#define RK_HPP

#include "base.hpp"

#include <pybind11/pybind11.h>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>

namespace BoxModel {
    class RKF45 : public OdeSolver {
        public:
            RKF45(pybind11::function f, double a, double b, double y0, double atol, double hmax, double hmin) :
                OdeSolver(f, a, b, y0, atol, hmax, hmin) 
                {

                    this->s = 6;
                    this->a = boost::numeric::ublas::matrix<double>(6,6);
                    this->b = boost::numeric::ublas::matrix<double>(2,6);
                    this->c = boost::numeric::ublas::vector<double>(6);

                    // Butcher Tableau
                    /*===========================================================================================================================================================================================*/
                    this->c(0) = 0.     ;/*|*/
                    this->c(1) = 1./4.  ;/*|*/ this->a(1,0) =  1./4.      ;
                    this->c(2) = 3./8.  ;/*|*/ this->a(2,0) =  3./32.     ; this->a(2,1) =  9./32.     ;
                    this->c(3) = 12./13.;/*|*/ this->a(3,0) =  1932./2197.; this->a(3,1) = -7200./2197.; this->a(3,2) =  7296./2197. ;
                    this->c(4) = 1.     ;/*|*/ this->a(4,0) =  439./216.  ; this->a(4,1) = -8.         ; this->a(4,2) =  3680./513.  ; this->a(4,3) = -845./4104.  ;
                    this->c(5) = 1./2.  ;/*|*/ this->a(5,0) = -8./27.     ; this->a(5,1) =  2.         ; this->a(5,2) = -3544./2565. ; this->a(5,3) =  1859./4104. ; this->a(5,4) = -11./40.;
                    /*===================//|//===================================================================================================================================================================*/
                                        /*|*/ this->b(0,0) =  1./360.    ; this->b(0,1) =  0.         ; this->b(0,2) = -128./4275.  ; this->b(0,3) = -2197./75240.; this->b(0,4) =  1./50. ; this->b(0,5) = 2./55.;
                                        /*|*/ this->b(1,0) =  25./216.   ; this->b(1,1) =  0.         ; this->b(1,2) =  1408./2565. ; this->b(1,3) =  2197./4104. ; this->b(1,4) = -1./5.  ; this->b(1,5) = 0.    ;  
                }

        protected:
            void solver_step();
            int s;
            boost::numeric::ublas::matrix<double> a, b;
            boost::numeric::ublas::vector<double> c;
             
    };
}

#endif /* RK_HPP */