# Plane strain plate under gravitational pull
#
# In this script we solve the linear elasticity problem on a unit square
# domain, clamped at the top boundary, and stretched under the influence of a
# vertical distributed load.

from nutils import mesh, function, solver, export, cli, testing
from nutils.expression_v2 import Namespace
import treelog as log
import numpy

def main(nelems: int, etype: str, btype: str, degree: int, poisson: float, direct: bool):
    '''
    Horizontally loaded linear elastic plate.

    .. arguments::

       nelems [24]
         Number of elements along edge.
       etype [triangle]
         Type of elements (square/triangle/mixed).
       btype [std]
         Type of basis function (std/spline), with availability depending on the
         configured element type.
       degree [2]
         Polynomial degree.
       poisson [.3]
         Poisson's ratio, nonnegative and strictly smaller than 1/2.
       direct [no]
         Use direct traction evaluation.
    '''

    domain, geom = mesh.unitsquare(nelems, etype)

    ns = Namespace()
    ns.δ = function.eye(domain.ndims)
    ns.x = geom
    ns.define_for('x', gradient='∇', normal='n', jacobians=('dV', 'dS'))
    ns.add_field(('u', 't'), domain.basis(btype, degree=degree), shape=(2,))
    ns.X_i = 'x_i + u_i'
    ns.λ = 1
    ns.μ = .5/poisson - 1
    ns.ε_ij = '.5 (∇_i(u_j) + ∇_j(u_i))'
    ns.σ_ij = 'λ ε_kk δ_ij + 2 μ ε_ij'
    ns.E = 'ε_ij σ_ij'
    ns.q_i = '-δ_i1'

    sqr = domain.boundary['top'].integral('u_k u_k dS' @ ns, degree=degree*2)
    cons = solver.optimize(('u',), sqr, droptol=1e-15)

    # solve for equilibrium configuration
    internal = domain.integral('E dV' @ ns, degree=degree*2)
    external = domain.integral('u_i q_i dV' @ ns, degree=degree*2)
    args = solver.optimize('u,', internal - external, constrain=cons)

    # evaluate tractions and net force
    if direct:
        ns.t_i = 'σ_ij n_j' # <-- this is an inadmissible boundary term
    else:
        external += domain.boundary['top'].integral('u_i t_i dS' @ ns, degree=degree*2)
        invcons = dict(t=numpy.choose(numpy.isnan(cons['u']), [numpy.nan, 0.]))
        args = solver.solve_linear(('t',), [(internal - external).derivative('u')], constrain=invcons, arguments=args)
    F = domain.boundary['top'].integrate('t_i dS' @ ns, degree=degree*2, arguments=args)
    log.user('total clamping force:', F)

    # visualize solution
    bezier = domain.sample('bezier', 3)
    X, E = bezier.eval(['X_i', 'E'] @ ns, **args)
    Xt, t = domain.boundary['top'].sample('bezier', 2).eval(['X_i', 't_i'] @ ns, **args)
    with export.mplfigure('energy.png') as fig:
        ax = fig.add_subplot(111, ylim=(-.2,1), aspect='equal')
        im = ax.tripcolor(*X.T, bezier.tri, E, shading='gouraud', rasterized=True, cmap='turbo')
        export.plotlines_(ax, X.T, bezier.hull, colors='k', linewidths=.1, alpha=.5)
        ax.quiver(*Xt.T, *t.T, clip_on=False)
        fig.colorbar(im)

    return cons, args

# If the script is executed (as opposed to imported), :func:`nutils.cli.run`
# calls the main function with arguments provided from the command line. For
# example, to keep with the default arguments simply run :sh:`python3
# elasticity.py`. To select mixed elements and quadratic basis functions add
# :sh:`python3 elasticity.py etype=mixed degree=2`.

if __name__ == '__main__':
    cli.run(main)

# Once a simulation is developed and tested, it is good practice to save a few
# strategic return values for regression testing. The :mod:`nutils.testing`
# module, which builds on the standard :mod:`unittest` framework, facilitates
# this by providing :func:`nutils.testing.TestCase.assertAlmostEqual64` for the
# embedding of desired results as compressed base64 data.

class test(testing.TestCase):

    @testing.requires('matplotlib')
    def test_default(self):
        cons, args = main(nelems=4, etype='square', btype='std', degree=1, poisson=.25, direct=False)
        with self.subTest('constraints'):
            self.assertAlmostEqual64(cons['u'], '''
                eNpjaGBAhSBAZTEAEKAUAQ==''')
        with self.subTest('displacement'):
            self.assertAlmostEqual64(args['u'], '''
                eNqT1yk8K6o35ay2PsO5ev3v5xiA4ItW1NlnOrVnX+l+PrtZV+Y8AxiEnGVgqATir0AsARbjuRp1Vupy
                7VmxS5/P+l6CqHt4ufDs64tTzl69wHCu8QLEPADBQyml''')
        with self.subTest('traction'):
            self.assertAlmostEqual64(args['t'], '''
                eNpjYEAF7Kd2mqMJMXAZiGKIMTCEYIh9O4+p7qfpTnMA/hUHAg==''')

    @testing.requires('matplotlib')
    def test_mixed(self):
        cons, args = main(nelems=4, etype='mixed', btype='std', degree=1, poisson=.25, direct=False)
        with self.subTest('constraints'):
            self.assertAlmostEqual64(cons['u'], '''
                eNpjaGDAD2EAmziRYgAgoBQB''')
        with self.subTest('solution'):
            self.assertAlmostEqual64(args['u'], '''
                eNoz1c0466vXfrZeJ+ystm7TWVl9lnPPdd+erdf/fG66rvR5Bijg0Ko4e0or+uwjjT9nHTVEweKOt2PO
                rrrcdjbj0uezTpdkwGK2l6afnXW14SznRZ5z+y5wgcUAJX0p8A==''')
        with self.subTest('traction'):
            self.assertAlmostEqual64(args['t'], '''
                eNpjYMAPXE7tNZ98ls8cXTzFOAVDbMfpz2boYqfMLpkDAGMuCY8=''')

    @testing.requires('matplotlib')
    def test_quadratic(self):
        cons, args = main(nelems=4, etype='square', btype='std', degree=2, poisson=.25, direct=False)
        with self.subTest('constraints'):
            self.assertAlmostEqual64(cons['u'], '''
                eNpjaGDAD0FgVA1eNQAj9kgB''')
        with self.subTest('solution'):
            self.assertAlmostEqual64(args['u'], '''
                eNolkL1KA0EUhcdI+sQnEFPtZma3tlBEQSwCJmskLFisvoAKu40hEAjaqkWaCKaIYCGSOhYiWlidmZ2Z
                /FQRXyBuirSim+yFr7jwcTn3MNPHkPr4ZU28WD0Ydorf2BV+bs/4o/0sSDzbpocT6qHB6ti0OniyIvxZ
                BT620uKL3YVz59tw0c67+KBV1FgbYzbFulXkI5YRJbazcLKGgyPTQTUfwKYtXNEJ3miZl2hW3Ob3Fw4h
                Dgg5iAliWjE/850TshJn2Vs40dDBw8DBZT+A1C1c6Am2dJl3dVac9pM7q0MXx30XG7qKhmojp6b4lEVu
                qIzoqiTP68DDvfZQU3W8yw4OZYQlWeA5mRZrKvlLD3yY2seyamJX9jAKU/wsrPDrcMbdMOnnH2mlmk4=''')
        with self.subTest('traction'):
            self.assertAlmostEqual64(args['t'], '''
                eNpjYMAPOk+qWxBQwvDW5L8pITUSZ9LMCamZaCBGUA0DQwBBNfnnCZvzwpiwe4RPE/ZXuZm6BQAVHQ1+''')

    @testing.requires('matplotlib')
    def test_poisson(self):
        cons, args = main(nelems=4, etype='square', btype='std', degree=1, poisson=.4, direct=False)
        with self.subTest('constraints'):
            self.assertAlmostEqual64(cons['u'], '''
                eNpjaGBAhSBAZTEAEKAUAQ==''')
        with self.subTest('solution'):
            self.assertAlmostEqual64(args['u'], '''
                eNqTNig6vcVwwekjRuJn5Iy1zzIAwQs999MdBmWn+w0Zz7QYpoPFGBisTzMw5AMx6xkGhniwmMRF99MV
                58tOF55jPFNzDqLu6fmi0z7nFpy2OSt+5tEZiHkAKRAl5A==''')
        with self.subTest('traction'):
            self.assertAlmostEqual64(args['t'], '''
                eNpjYEAF/Sc+maMJMdw0emzGgAFiMdSpn8VUV2j+yRwAoCAJFw==''')

# example:tags=elasticity
