from nutils import function, export, mesh, solver, testing, cli
import numpy as np
def shapes(i, j, k):
    domain, geom0 = mesh.rectilinear([i-1, j-1, k-1], periodic=[0])
    # Knot vector and knot multiplicities
    kv = [np.arange(1,i+1), np.arange(0,j), np.arange(0,k)]
    km = [[2]*i, [2]*j, [2]*k]
    bsplinebasis = domain.basis('spline',
                                degree=(2, 1, 1),
                                knotmultiplicities=km,
                                knotvalues=kv,
                                periodic=[0])
    print(geom0[:2])
    print(domain[:2])
    print(bsplinebasis[:2])


shapes(5,2,2)
"""for i in range(2,10):
    for j in range(2,5):
        shapes(i, j, 2)
    print("yes")
shapes(6, 4, 6)"""