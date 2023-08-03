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
    print(domain.spaces)
    print(domain.ndims)
    print(domain.references)
    print(bsplinebasis.points_shape)


def get_cylinder(inner_radius, outer_radius, height, nrefine=None):
    """Creates a periodic hollow cylinder of with defined inner and outer
    radius along the z axis with given height.

    Parameters
    ----------
    inner_radius : float
        Inner radius
    outer_radius : float
        Outer radius
    height : float
        Length along z-axis

    Returns
    -------
    nutils.topology, nutils._Wrapper, nutils.function
        Domain, geometry and nurbs basis of nutils functions"""
    
    domain, geom0 = mesh.rectilinear([4, 1, 1], periodic=[0])

    # Knot vector and knot multiplicities
    kv = [[0, 1, 2, 3, 4], [0, 1], [0, 1]]
    km = [[2, 2, 2, 2, 2], [2, 2], [2, 2]]
    print("yes")
    bsplinebasis = domain.basis('spline',
                                degree=(2, 1, 1),
                                knotmultiplicities=km,
                                knotvalues=kv,
                                periodic=[0])
    
    """i = 5
    j = 2
    k = 2
    domain, geom0 = mesh.rectilinear([i-1, j-1, k-1], periodic=[0])
    # Knot vector and knot multiplicities
    kv = [np.arange(1,i+1), np.arange(0,j), np.arange(0,k)]
    km = [[2]*i, [2]*j, [2]*k]
    print(domain)
    print(geom0)"""
    bsplinebasis = domain.basis('spline',
                                degree=(2, 1, 1),
                                knotmultiplicities=km,
                                knotvalues=kv,
                                periodic=[0]
                                )
    cps = np.array([[inner_radius, 0, 0], [outer_radius, 0, 0],
                    [inner_radius, 0, height], [outer_radius, 0, height],
                    [inner_radius, -inner_radius, 0],
                    [outer_radius, -outer_radius, 0],
                    [inner_radius, -inner_radius, height],
                    [outer_radius, -outer_radius,
                     height], [0, -inner_radius, 0], [0, -outer_radius, 0],
                    [0, -inner_radius, height], [0, -outer_radius, height],
                    [-inner_radius, -inner_radius, 0],
                    [-outer_radius, -outer_radius, 0],
                    [-inner_radius, -inner_radius, height],
                    [-outer_radius, -outer_radius, height],
                    [-inner_radius, 0, 0], [-outer_radius, 0, 0],
                    [-inner_radius, 0, height], [-outer_radius, 0, height],
                    [-inner_radius, inner_radius, 0],
                    [-outer_radius, outer_radius, 0],
                    [-inner_radius, inner_radius, height],
                    [-outer_radius, outer_radius, height],
                    [0, inner_radius, 0], [0, outer_radius, 0],
                    [0, inner_radius, height], [0, outer_radius, height],
                    [inner_radius, inner_radius, 0],
                    [outer_radius, outer_radius, 0],
                    [inner_radius, inner_radius, height],
                    [outer_radius, outer_radius, height]])

    #cps = np.random.rand(32,3)
    #print(bsplinebasis)
   
    #cps =  custom_shape.generate_cps()
    

    controlweights = np.tile(np.repeat([1., 1 / np.sqrt(2)], 4), 4)
    

    # Create nurbsbasis and geometry
    weightfunc = bsplinebasis.dot(controlweights)
    nurbsbasis = bsplinebasis * controlweights / weightfunc
    geom = (nurbsbasis[:, np.newaxis] * cps).sum(0)
    # Refine domain nrefine times
    if nrefine:
        domain = domain.refine(nrefine)
        bsplinebasis = domain.basis('spline', degree=2)
        controlweights = domain.project(weightfunc,
                                        onto=bsplinebasis,
                                        geometry=geom0,
                                        ischeme='gauss9')
        nurbsbasis = bsplinebasis * controlweights / weightfunc

    return domain, geom, nurbsbasis

def parameters():
    # Create a hollow cylinder as geometry
    domain, geom, nurbsbasis = get_cylinder(inner_radius=1.,
                                            outer_radius=1.5,
                                            height=2.0,
                                            nrefine=None)
    print(domain.spaces)
    print(domain.ndims)
    print(domain.references)
    print(np.array(geom))
    for i in np.array(geom):
        print(i)
    print(geom.ndim)
    print(nurbsbasis.broadcasted_arrays)
    print(nurbsbasis.ndim)

def cube_ctr():
    points=[-1, 0, 1]
    cps = []
    for i in points:
        for j in points:
            for k in points:
                cps.append([i, j, k])

    return cps
print(cube_ctr())
