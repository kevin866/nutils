from nutils import function, export, mesh, solver, testing, cli
from nutils.expression_v2 import Namespace
import treelog
import numpy as np
from matplotlib.pyplot import Normalize
from matplotlib.cm import coolwarm, ScalarMappable


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
        Domain, geometry and nurbs basis of nutils functions
    """

    domain, geom0 = mesh.rectilinear([4, 1, 1], periodic=[0])

    # Knot vector and knot multiplicities
    kv = [[0, 1, 2, 3, 4], [0, 1], [0, 1]]
    km = [[2, 2, 2, 2, 2], [2, 2], [2, 2]]

    bsplinebasis = domain.basis('spline',
                                degree=(2, 1, 1),
                                knotmultiplicities=km,
                                knotvalues=kv,
                                periodic=[0])

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


def main(nrefine=1,
         nelems: int = 10,
         etype: str = 'square',
         btype: str = 'std',
         degree: int = 1):
    """The function simulates the cooling of a cylinder with non-homogeneous
    temperature. The solution of the heat equation with Dirichlet boundary
    conditions serves as initial values for a thermoelastic problem.

    Parameters
    ----------
    nrefine : int, optional
        Number of refinements of the domain, by default 1
    poisson : float, optional
        poisson ration of the material, by default 0.3
    diffusivity : float, optional
        diffusivity of the material, by default 0.01
    thermal_expansion : float, optional
        thermal expansion of the material, by default 0.01
    timestep : float, optional
        timestep in the solution of the heat problem, by default 0.1
    endtime : float, optional
        endtime of the heat problem, by default 0.2
    """

    ns = Namespace()

    # Create a hollow cylinder as geometry
    # domain, geom, nurbsbasis = get_cylinder(inner_radius=1.,
    #                                         outer_radius=1.5,
    #                                         height=2.0,
    #                                         nrefine=nrefine)
    domain, ns.x0 = mesh.rectilinear([np.linspace(0, 1, nelems + 1)] * 3)
    ns.define_for('x0', jacobians=('dV0',))

    # Define deformed geometry `ns.x` in terms of a spline basis and argument `x`.
    basis = domain.basis('spline', degree=degree)
    ns.add_field('x', basis, shape=(domain.ndims,))
    ns.define_for('x', gradient='∇', jacobians=('dV', 'dS'))

    # Initialize argument `x` by projecting the initial geometry `cube` onto the basis.
    args = solver.optimize('x,', domain.integral('(x_i - x0_i) (x_i - x0_i) dV0' @ ns, degree=2 * degree))

    # Deform the geometry by adding a random offset to argument `x`.
    rng = np.random.default_rng(seed=0) # `seed=0` for reproducibility

    args['x'] = args['x'] + rng.normal(scale=0.1, size=args['x'].shape)


    # domain, geom = mesh.rectilinear([np.linspace(0, 1, nelems + 1)] * 3)
    # ns.x = geom
    # print(domain.boundary)

    ns.define_for('x', gradient='∇', normal='n', jacobians=('dV', 'dS'))
    ns.add_field(('u', 'v'), domain.basis(btype, degree=degree))

    # We are now ready to implement the Laplace equation. In weak form, the
    # solution is a scalar field `u` for which ∫_Ω ∇v·∇u - ∫_Γn v f = 0 ∀ v.
    print(dir(domain.boundary))
    res = domain.integral('∇_i(v) ∇_i(u) dV' @ ns, degree=degree*2)
    res -= domain.boundary['right'].integral('v cos(1) cosh(x_1) dS' @ ns, degree=degree*2)

    # The Dirichlet constraints are set by finding the coefficients that
    # minimize the error ∫_Γd (u - u_d)^2. The resulting `cons` dictionary
    # holds numerical values for all the entries of the function argument `u`
    # that contribute (up to `droptol`) to the minimization problem. All
    # remaining entries are set to `NaN`, signifying that these degrees of
    # freedom are unconstrained.

    sqr = domain.boundary['left'].integral('u^2 dS' @ ns, degree=degree*2)
    sqr += domain.boundary['top'].integral('(u - cosh(1) sin(x_0))^2 dS' @ ns, degree=degree*2)
    cons = solver.optimize('u,', sqr, droptol=1e-15, arguments=args)

    # The unconstrained entries of `u` are to be such that the residual
    # evaluates to zero for all possible values of `v`. The resulting array `u`
    # matches `cons` in the constrained entries.

    args = solver.solve_linear('u:v', res, constrain=cons, arguments=args)

    # Once all arguments are establised, the corresponding solution can be
    # vizualised by sampling values of `ns.u` along with physical coordinates
    # `ns.x`, with the solution vector provided via keyword arguments. The
    # sample members `tri` and `hull` provide additional inter-point
    # information required for drawing the mesh and element outlines.
    bezier = domain.sample('bezier', 9)
    xsmp, usmp = bezier.eval(['x_i', 'u'] @ ns, **args)
    
    # export.triplot('solution.png', xsmp, usmp, tri=bezier.tri, hull=bezier.hull)
    export.vtk('deformed_cylinder', bezier.tri,xsmp, u = usmp)
    print('yse')
    X = xsmp
    with export.mplfigure('displacement.png') as fig:
        ax = fig.add_subplot(111, projection='3d')

        meanU = np.array([np.mean(usmp[t]) for t in bezier.tri])
        norm = Normalize(np.min(meanU), np.max(meanU))
        surf = ax.plot_trisurf(X[:, 0], X[:, 1], X[:, 2], triangles=bezier.tri)
        surf.set_fc(coolwarm(norm(meanU)))

        cbar = fig.colorbar(ScalarMappable(cmap=coolwarm, norm=norm), ax=ax)
        cbar.set_label('Displacement')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
    # To confirm that our computation is correct, we use our knowledge of the
    # analytical solution to evaluate the L2-error of the discrete result.

    # err = domain.integral('(u - sin(x_0) cosh(x_1))^2 dV' @ ns, degree=degree*2).eval(**args)**.5
    # treelog.user('L2 error: {:.2e}'.format(err))

    return cons['u'], args['u']

class test(testing.TestCase):

    def test_simple(self):
        cons, u, err = main(nelems=4)
        with self.subTest('constraints'):
            self.assertAlmostEqual64(cons, '''
                eNrbKPv1QZ3ip9sL1BgaILDYFMbaZwZj5ZnDWNfNAeWPESU=''')
        with self.subTest('left-hand side'):
            self.assertAlmostEqual64(u, '''
                eNoBMgDN/7Ed9eB+IfLboCaXNKc01DQaNXM14jXyNR82ZTa+NpI2oTbPNhU3bjf7Ngo3ODd+N9c3SNEU
                1g==''')
        with self.subTest('L2-error'):
            self.assertAlmostEqual(err, 1.63e-3, places=5)

    def test_spline(self):
        cons, u, err = main(nelems=4, btype='spline', degree=2)
        with self.subTest('constraints'):
            self.assertAlmostEqual64(cons, '''
                eNqrkmN+sEfhzF0xleRbDA0wKGeCYFuaIdjK5gj2aiT2VXMAJB0VAQ==''')
        with self.subTest('left-hand side'):
            self.assertAlmostEqual64(u, '''
                eNqrkmN+sEfhzF0xleRbrsauxsnGc43fGMuZJJgmmNaZ7jBlN7M08wLCDLNFZh/NlM0vmV0y+2CmZV5p
                vtr8j9kfMynzEPPF5lfNAcuhGvs=''')
        with self.subTest('L2-error'):
            self.assertAlmostEqual(err, 8.04e-5, places=7)

    def test_mixed(self):
        cons, u, err = main(nelems=4, etype='mixed', degree=2)
        with self.subTest('constraints'):
            self.assertAlmostEqual64(cons, '''
                eNorfLZF2ucJQwMC3pR7+QDG9lCquAtj71Rlu8XQIGfC0FBoiqweE1qaMTTsNsOvRtmcoSHbHL+a1UD5
                q+YAxhcu1g==''')
        with self.subTest('left-hand side'):
            self.assertAlmostEqual64(u, '''
                eNorfLZF2ueJq7GrcYjxDJPpJstNbsq9fOBr3Gh8xWS7iYdSxd19xseMP5hImu5UZbv1xljOxM600DTW
                NN/0k2mC6SPTx6Z1pnNMGc3kzdaaPjRNMbMyEzWzNOsy223mBYRRZpPNJpktMks1azM7Z7bRbIXZabNX
                ZiLmH82UzS3Ns80vmj004za/ZPYHCD+Y8ZlLmVuYq5kHm9eahwDxavPF5lfNAWFyPdk=''')
        with self.subTest('L2-error'):
            self.assertAlmostEqual(err, 1.25e-4, places=6)


# If the script is executed (as opposed to imported), `nutils.cli.run` calls
# the main function with arguments provided from the command line. For example,
# to keep with the default arguments simply run `python3 laplace.py`. To select
# mixed elements and quadratic basis functions add `python3 laplace.py
# etype=mixed degree=2`.

if __name__ == '__main__':
    from nutils import cli
    cli.run(main)


# example:tags=Laplace
