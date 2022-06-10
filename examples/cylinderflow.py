#! /usr/bin/env python3
#
# In this script we solve the Navier-Stokes equations around a cylinder, using
# the same Raviart-Thomas discretization as in
# :ref:`examples/drivencavity-compatible.py` but in curvilinear coordinates.
# The mesh is constructed such that all elements are shape similar, growing
# exponentially with radius such that the artificial exterior boundary is
# placed at large (configurable) distance.

from nutils import mesh, function, solver, util, export, cli, testing, numeric
from nutils.expression_v2 import Namespace
import itertools
import numpy
import treelog


class PostProcessor:

    def __init__(self, topo, ns, region=4., aspect=16/9, figscale=7.2, spacing=.05, pstep=.1, vortlim=20):
        self.ns = ns
        self.figsize = aspect * figscale, figscale
        self.bbox = numpy.array([[-.5, aspect-.5], [-.5, .5]]) * region
        self.bezier = topo.select(numpy.minimum(*(ns.x-self.bbox[:,0])*(self.bbox[:,1]-ns.x))).sample('bezier', 5)
        self.spacing = spacing
        self.pstep = pstep
        self.vortlim = vortlim
        self.t = 0.
        self.initialize_xgrd()
        self.topo = topo

    def initialize_xgrd(self):
        self.orig = numeric.floor(self.bbox[:,0] / (2*self.spacing)) * 2 - 1
        nx, ny = numeric.ceil(self.bbox[:,1] / (2*self.spacing)) * 2 + 2 - self.orig
        self.vacant = numpy.hypot(
            self.orig[0] + numpy.arange(nx)[:,numpy.newaxis],
            self.orig[1] + numpy.arange(ny)) > .5 / self.spacing
        self.xgrd = (numpy.stack(self.vacant[1::2,1::2].nonzero(), axis=1) * 2 + self.orig + 1) * self.spacing

    def regularize_xgrd(self):
        # use grid rounding to detect and remove oldest points that have close
        # neighbours and introduce new points into vacant spots
        keep = numpy.zeros(len(self.xgrd), dtype=bool)
        vacant = self.vacant.copy()
        for i, ind in enumerate(numeric.round(self.xgrd / self.spacing) - self.orig): # points are ordered young to old
            if all(ind >= 0) and all(ind < vacant.shape) and vacant[tuple(ind)]:
                vacant[tuple(ind)] = False
                keep[i] = True
        roll = numpy.arange(vacant.ndim)-1
        for _ in roll: # coarsen all dimensions using 3-point window
            vacant = numeric.overlapping(vacant.transpose(roll), axis=0, n=3)[::2].all(1)
        newpoints = numpy.stack(vacant.nonzero(), axis=1) * 2 + self.orig + 1
        self.xgrd = numpy.concatenate([newpoints * self.spacing, self.xgrd[keep]], axis=0)

    def __call__(self, args):
        x, p, ω = self.bezier.eval(['x_i', 'p', 'ω'] @ self.ns, **args)
        grid0 = numpy.log(numpy.hypot(*self.xgrd.T) / .5)
        grid1 = numpy.arctan2(*self.xgrd.T) % (2 * numpy.pi)
        ugrd = self.topo.locate(self.ns.grid, numpy.stack([grid0, grid1], axis=1), eps=1, tol=1e-5).eval(self.ns.u, **args)
        with export.mplfigure('flow.png', figsize=self.figsize) as fig:
            ax = fig.add_axes([0, 0, 1, 1], yticks=[], xticks=[], frame_on=False, xlim=self.bbox[0], ylim=self.bbox[1])
            ax.tripcolor(*x.T, self.bezier.tri, ω, shading='gouraud', cmap='seismic').set_clim(-self.vortlim, self.vortlim)
            ax.tricontour(*x.T, self.bezier.tri, p, numpy.arange(numpy.min(p)//1, numpy.max(p), self.pstep), colors='k', linestyles='solid', linewidths=.5, zorder=9)
            export.plotlines_(ax, x.T, self.bezier.hull, colors='k', linewidths=.1, alpha=.5)
            ax.quiver(*self.xgrd.T, *ugrd.T, angles='xy', width=1e-3, headwidth=3e3, headlength=5e3, headaxislength=2e3, zorder=9, alpha=.5, pivot='tip')
            ax.plot(0, 0, 'k', marker=(3, 2, -self.t*self.ns.rotation.eval()*180/numpy.pi-90), markersize=20)
        dt = self.ns.dt.eval()
        self.t += dt
        self.xgrd += ugrd * dt
        self.regularize_xgrd()


def main(nelems: int, degree: int, reynolds: float, uwall: float, timestep: float, extdiam: float, endtime: float):
    '''
    Flow around a cylinder.

    .. arguments::

       nelems [63]
         Element size expressed in number of elements along the cylinder wall.
         All elements have similar shape with approximately unit aspect ratio,
         with elements away from the cylinder wall growing exponentially. Use
         an odd number to break symmetry and promote early bifurcation.
       degree [3]
         Polynomial degree for velocity space; the pressure space is one degree
         less.
       reynolds [1000]
         Reynolds number, taking the cylinder diameter as characteristic length.
       uwall [0]
         Cylinder wall velocity.
       timestep [.04]
         Time step
       extdiam [50]
         Target exterior diameter; the actual domain size is subject to integer
         multiples of the configured element size.
       endtime [inf]
         Stopping time.
    '''

    elemangle = 2 * numpy.pi / nelems
    melems = round(numpy.log(extdiam) / elemangle)
    treelog.info('creating {}x{} mesh, outer radius {:.2f}'.format(melems, nelems, .5*numpy.exp(elemangle*melems)))
    domain, geom = mesh.rectilinear([melems, nelems], periodic=(1,))
    domain = domain.withboundary(inner='left', inflow=domain.boundary['right'][nelems//2:])

    ns = Namespace()
    ns.δ = function.eye(domain.ndims)
    ns.Σ = function.ones([domain.ndims])
    ns.ε = function.levicivita(2)
    ns.uinf_i = 'δ_i0' # unit horizontal flow
    ns.Re = reynolds
    ns.grid = geom * elemangle
    ns.x_i = '.5 exp(grid_0) (sin(grid_1) δ_i0 + cos(grid_1) δ_i1)' # polar coordinates
    ns.define_for('x', gradient='∇', normal='n', jacobians=('dV', 'dS'))
    J = ns.x.grad(geom)
    detJ = function.determinant(J)
    ns.add_field(('u', 'u0', 'v'), function.matmat(function.vectorize([
        domain.basis('spline', degree=(degree, degree-1), removedofs=((0,), None)),
        domain.basis('spline', degree=(degree-1, degree))]), J.T) / detJ)
    ns.add_field(('p', 'q'), domain.basis('spline', degree=degree-1) / detJ)
    ns.dt = timestep
    ns.DuDt_i = '(u_i - u0_i) / dt + ∇_j(u_i) u_j' # material derivative
    ns.σ_ij = '(∇_j(u_i) + ∇_i(u_j)) / Re - p δ_ij'
    ns.ω = 'ε_ij ∇_j(u_i)'
    ns.N = 10 * degree / elemangle  # Nitsche constant based on element size = elemangle/2
    ns.nitsche_i = '(N v_i - (∇_j(v_i) + ∇_i(v_j)) n_j) / Re'
    ns.rotation = uwall / .5
    ns.uwall_i = 'rotation ε_ij x_j' # clockwise positive rotation

    sqr = domain.boundary['inflow'].integral('Σ_i (u_i - uinf_i)^2 dS' @ ns, degree=degree*2)
    cons = solver.optimize('u,', sqr, droptol=1e-15) # constrain inflow boundary to unit horizontal flow

    sqr = domain.integral('(.5 Σ_i (u_i - uinf_i)^2 - ∇_k(u_k) p) dV' @ ns, degree=degree*2)
    args = solver.optimize('u,p', sqr, constrain=cons) # set initial condition to potential flow

    res = domain.integral('(v_i DuDt_i + ∇_j(v_i) σ_ij + q ∇_k(u_k)) dV' @ ns, degree=degree*3)
    res += domain.boundary['inner'].integral('(nitsche_i (u_i - uwall_i) - v_i σ_ij n_j) dS' @ ns, degree=degree*2)
    div = numpy.sqrt(domain.integral('∇_k(u_k)^2 dV' @ ns, degree=2)) # L2 norm of velocity divergence

    postprocess = PostProcessor(domain, ns)

    steps = treelog.iter.fraction('timestep', range(round(endtime / timestep))) if endtime < float('inf') \
       else treelog.iter.plain('timestep', itertools.count())

    for _ in steps:
        treelog.info(f'velocity divergence: {div.eval(**args):.0e}')
        args['u0'] = args['u']
        args = solver.newton('u:v,p:q', residual=res, arguments=args, constrain=cons).solve(1e-10)
        postprocess(args)

    return args

# If the script is executed (as opposed to imported), :func:`nutils.cli.run`
# calls the main function with arguments provided from the command line.


if __name__ == '__main__':
    cli.run(main)

# Once a simulation is developed and tested, it is good practice to save a few
# strategic return values for regression testing. The :mod:`nutils.testing`
# module, which builds on the standard :mod:`unittest` framework, facilitates
# this by providing :func:`nutils.testing.TestCase.assertAlmostEqual64` for the
# embedding of desired results as compressed base64 data.


class test(testing.TestCase):

    def test_rot0(self):
        args = main(nelems=6, degree=3, reynolds=100, uwall=0, timestep=.1, extdiam=50, endtime=.1)
        with self.subTest('velocity'):
            self.assertAlmostEqual64(args['u'], '''
                eNoBkABv//AzussRy7rL8DNVNU42sskxyLLJTjbPN7Q4SscGxkrHtDj9ObM6SMXmw0jFszofPFU8nsNk
                wp7DVTyqPS49usKawbrCLj2APuHJi8hHyrk1dTcfNmbJJMhDyb023DeaNiPItMYoyNg3TDndNwnGv8QO
                xvI5QTv3ORTErsIqxNY7Uj3sO8XCY8H1wgs9nT47Pc/9SG4=''')
        with self.subTest('pressure'):
            self.assertAlmostEqual64(args['p'], '''
                eNoBSAC3/7w0bzXBzG81vDRXytwzezW0y3s13DOXyYfOxzVVM8c1h87LyJTJ3DezN9w3lMkBxzTIDDgz
                Ogw4NMhAxu42Ij1DxCI97jZ+wirgIsM=''')

    def test_rot1(self):
        args = main(nelems=6, degree=3, reynolds=100, uwall=.5, timestep=.1, extdiam=50, endtime=.1)
        with self.subTest('velocity'):
            self.assertAlmostEqual64(args['u'], '''
                eNoBkABv//czw8sRy7HL6TNVNU82tckxyLDJTTbPN7Q4SscGxkrHszj9ObM6SMXmw0jFszofPFU8nsNk
                wp7DVTyqPS49usKawbrCLj2APrnJdMgEym01XDf1NXHJKshPyck24jelNiHIs8YnyNc3SznaNwnGv8QO
                xvI5QTv4ORTErcIqxNY7Uj3sO8XCY8H1wgs9nT47PdHgSI0=''')
        with self.subTest('pressure'):
            self.assertAlmostEqual64(args['p'], '''
                eNoBSAC3/+M0kjXDzEs1kjRXyvszijW0y2w1ujOXyV0tAzZXM4I1Dc3LyA7KDTizN6Y3MckBxybJpDgz
                OjE3j8dAxr84Pz1DxAQ9I8p9wpetHyk=''')
