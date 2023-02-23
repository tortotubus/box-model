#include "rkf45.hpp"

#include <vector>
#include <iostream>

void BoxModel::RKF45::solver_step() 
{   
    double safety = 0.9;
    double min_factor = 0.2;
    double max_factor = 10.;

    std::valarray<double> *k = new std::valarray<double>[this->n_stages];
    for (int z = 0; z < this->n_stages; z++) { k[z] = std::valarray<double>(0.,this->n); }

    for (int i = 0; i < this->n_stages; i++)
    {
        std::valarray<double> dy(0., this->n);

        if (i > 0) 
        {
            for (int j = 0; j < i; j++) 
            { 
                dy += this->a[i][j] * k[j]; 
            }
        }

        k[i] = fprime_wrapped(this->t + this->c[i] * this->h, this->y + dy * this->h);
    }

    // Calculate y_new and f_new
    std::valarray<double> y_new(0., this->n);
    std::valarray<double> f_new(this->f);
    std::valarray<double> e_new(0., this->n);

    for (int i = 0; i < this->n_stages; i++) 
    { 
        y_new += this->b[i] * k[i]; 
        e_new += this->e[i] * k[i]; 
    }

    std::valarray<double> scale(0., this->n);
    for (int i = 0; i < this->n; i++) { scale[i] = std::max(std::abs(this->y[i]), std::abs(y_new[i])); }
    scale *= rtol;
    scale += atol;

    double error_norm = std::sqrt(std::pow((e_new*this->h)/scale, 2).sum() / this->n);

    std::cout << "error_norm:" << error_norm << " ";
    std::cout << "Scale: [" << scale[0] << "], ";
    std::cout << "y:[" << this->y[0] << "]";
    std::cout << "y_new:[" << this->y[0]+y_new[0]*this->h << "]";
    std::cout << "Step size: " << this->h;
    std::cout << std::endl;

    double error_exponent = -1./((double)this->error_estimator_order+1.);

    if (error_norm < 1) // Step is good
    {
        this->step_accepted = 1;
        this->step_rejected = 0;

        this->y += y_new * this->h;
        this->f = f_new;
        this->t_old = this->t;
        this->t += this->h;

        double factor;

        if (error_norm == 0) 
            factor = max_factor;
        else
            factor = std::min(max_factor, safety * std::pow(error_norm, error_exponent));

        if (this->step_rejected)
            factor = std::min(1., factor);

        this->h *= factor;

    } 
    else
    {
        this->h *= std::max(min_factor, safety * std::pow(error_norm, error_exponent));
        this->step_accepted = 0;
        this->step_rejected = 1;
    }




/*  std::valarray<double> error_estimate(0., this->n);
    for (int i = 0; i < this->n_stages; i++) { error_estimate += this->e[i] * k[i]; }
    error_estimate = (1.0/(this->h)) * std::abs(error_estimate);
    double error = std::sqrt(std::pow(error_estimate, 2).sum());

    //std::cout << "error: " << error << std::endl;

    // Approximation accepted
    if (error <= this->atol) 
    { 
        for (int i = 0; i < this->n_stages; i++) { this->y += this->b[i] * k[i]; }
        this->t += this->h;
    }

    double delta = 0.84 * std::pow((this->atol/error), 0.25);

    if ( delta <= 0.1 ) { this->h *= 0.1; }
    else if ( delta >= 4 ) { this->h *= 4.0; }
    else { this->h *= delta; }

    if ( h > this->h_max ) { this->h = this->h_max; }

    if ( this->t >= this->t_end ) { this->complete = 1; }  // Reached end of interval
    else if ( this->t + this->h > this->t_end ) { this->h = this->t_end - this->t; }
    else if ( this->h < this->h_min ) { this->failure = 1; } // Minimum step size exceeded
*/
    // see pg 167 Harier
    
 }

