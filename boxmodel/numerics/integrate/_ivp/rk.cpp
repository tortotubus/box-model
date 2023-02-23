#include "rk.hpp"

#include <vector>
#include <iostream>

void BoxModel::AdaptiveRungeKutta::solver_step() 
{   
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


    this->y += y_new * this->h;
    this->f = f_new;
    this->t_old = this->t;
    this->t += this->h;

 }

