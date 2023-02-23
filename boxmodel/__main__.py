from .run_box_model import (model)
from .test_numerics import (test_rfk45)
from .test_numerics_vectorized import (test_numerics_vectorized)

if __name__ == '__main__':
    #model()
    test_rfk45()
    test_numerics_vectorized()
