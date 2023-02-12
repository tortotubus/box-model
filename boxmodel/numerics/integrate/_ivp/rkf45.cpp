#include "rkf45.hpp"
#include <cmath>

void BoxModel::RKF45::solver_step() 
{        
    double k[6] = {0.};

    for (int i = 0; i < this->s; i++)
    {
        double a = this->t + this->c[i] * this->h;
        double b = this->y;

        if (i > 0) 
        {
            for (int j = 0; j < i; j++) 
            { 
                b += this->a[i][j] * k[j]; 
            }
        }
        
        k[i] = this->h * f(a, b);
    }

    double error_estimate = 0;
    for (int i = 0; i < this->s; i++) { error_estimate += this->b[0][i] * k[i]; }
    error_estimate = (1.0/(this->h)) * std::abs(error_estimate);

    // Approximation accepted
    if (error_estimate <= this->atol) 
    { 
        for (int i = 0; i < this->s; i++) { this->y += this->b[1][i] * k[i]; }
        this->t += this->h;
    }

    double delta = 0.84 * std::pow((this->atol/error_estimate), 0.25);

    if ( delta <= 0.1 ) { this->h *= 0.1; }
    else if ( delta >= 4 ) { this->h *= 4.0; }
    else { this->h *= delta; }

    if ( h > this->h_max ) { this->h = this->h_max; }

    if ( this->t >= this->t_end ) { this->complete = 1; }  // Reached end of interval
    else if ( this->t + this->h > this->t_end ) { this->h = this->t_end - this->t; }
    else if ( this->h < this->h_min ) { this->failure = 1; } // Minimum step size exceeded

}