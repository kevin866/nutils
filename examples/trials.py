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

def cube_ctr(num):
    points=[-1, 0, 1]
    cps = []
    for i in points:
        for j in points:
            for k in points:
                cps.append([i, j, k])
    for i in range(num-27):
        cps.append(np.random.random(size=3))

    return cps
#print(cube_ctr(32))

def coe_p(p, alpha):
    p = (1-alpha)*p+alpha*np.random.uniform(low=-1, high=1, size=(32,3))
    return p

import numpy
from nutils import export, function, mesh, solver
from nutils.expression_v2 import Namespace
from matplotlib.pyplot import Normalize
from matplotlib.cm import coolwarm, ScalarMappable
def irregular_cube():
    nelems = 4
    stddev = 0.15
    degree = 3

    # Create a unit cube.
    topo, cube = mesh.rectilinear([numpy.linspace(0, 1, nelems + 1)] * 3)

    # Define deformed geometry `geom` in terms of a spline basis and argument `geom`.
    basis = topo.basis('spline', degree=degree)
    geom = basis @ function.Argument('geom', shape=(len(basis), 3))

    # Initialize argument `geom` by projecting the initial geometry `cube` onto the basis.
    args = solver.optimize('geom,', topo.integral(numpy.sum((cube - geom)**2) * function.J(cube), degree=2 * degree))

    # Deform the geometry by adding a random offset to argument `geom`.
    rng = numpy.random.default_rng(seed=0) # `seed=0` for reproducibility
    args['geom'] = args['geom'] + rng.normal(scale=stddev, size=args['geom'].shape)
    print(args['geom'].shape)
    print(geom)

    # Plot the surface of the cube.
    smpl = topo.boundary.sample('bezier', 5)
    x = smpl.eval(geom, **args)

    #export.triplot('surface.png', x, hull=smpl.hull)
    integration_degree = 4
    
    return topo, basis, integration_degree, geom



def simple_cube():
    nelems = 4
    shape = [np.linspace(0, 1, nelems + 1)]*3
    #shape = shape + np.random.random((3,5))
    domain, geom = mesh.rectilinear(shape, periodic=[4])
    print(geom)
    
    degree = 3
    basis = domain.basis('spline', degree=degree)
    integration_degree = 4
    return domain, basis, integration_degree, geom
