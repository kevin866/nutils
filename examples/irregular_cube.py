from nutils import function, export, mesh, solver, testing, cli
from nutils.expression_v2 import Namespace
import treelog
import numpy as np
from matplotlib.pyplot import Normalize
from matplotlib.cm import coolwarm, ScalarMappable


def main(nelems=4,
         degree=3,
         stddev=0.25,
         nrefine=1,
         poisson=0.3,
         diffusivity=0.01,
         thermal_expansion=0.01,
         timestep=0.1,
         endtime=0.2):
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

    # Create a unit cube.
    domain, ns.x0 = mesh.rectilinear([np.linspace(0, 1, nelems + 1)] * 3)
    ns.define_for('x0', jacobians=('dV0',))

    # Define deformed geometry `ns.x` in terms of a spline basis and argument `x`.
    basis = domain.basis('spline', degree=degree)
    ns.add_field('x', basis, shape=(domain.ndims,))
    ns.define_for('x', gradient='∇', jacobians=('dV', 'dS', 'dL'))

    # Initialize argument `x` by projecting the initial geometry `cube` onto the basis.
    args = solver.optimize('x,', domain.integral('(x_i - x0_i) (x_i - x0_i) dV0' @ ns, degree=2 * degree))

    # Deform the geometry by adding a random offset to argument `x`.
    rng = np.random.default_rng(seed=0) # `seed=0` for reproducibility
    stddev = [0.1, 0.1, 0.1]
    comp = np.array([1.0-i for i in stddev])
    args['x'] = np.multiply(np.array(args['x']),comp) + rng.normal(loc=[2.0,1.0,0.5], scale=stddev, size=args['x'].shape)


    # Heat equation
    ns.add_field(('T', 'S'), basis)
    ns.k = diffusivity
    ns.boundaryT = 5.0

    # Weak form of the heat equation
    tres = domain.integral('k ∇_i(S) ∇_i(T) dV' @ ns, degree=2 * degree)
    tinertia = domain.integral('S T dV' @ ns, degree=2 * degree)

    # Enforce temperature on the boundary, which increases with height
    sqr = domain.boundary.integral('(T - boundaryT x_2)^2 dS' @ ns,
                                   degree=2 * degree)
    tcons = solver.optimize('T,', sqr, droptol=1e-12, arguments=args)

    boundary_constrains = domain.boundary.project(fun='boundaryT x_2' @ ns,
                                                  onto=basis,
                                                  geometry=ns.x,
                                                  ischeme='gauss9',
                                                  arguments=args)
    args['T'] = domain.project(fun=0.,
                               onto=basis,
                               geometry=ns.x,
                               constrain=boundary_constrains,
                               ischeme='gauss9',
                               arguments=args)
    solution = []

    # Solve the heat problem
    with treelog.iter.plain(
            'timestep',
            solver.impliciteuler('T:S,',
                                 tres,
                                 tinertia,
                                 timestep=timestep,
                                 constrain=tcons,
                                 arguments=args)) as steps:
        for itime, args in enumerate(steps):

            args['timestep'] = itime * timestep
            solution.append(args.copy())

            if itime * timestep >= endtime:
                break

    # Elasticity problem
    ns.add_field(('u', 'v'), basis, shape=(domain.ndims,))

    ns.lmbda = 2 * poisson
    ns.mu = 1 - poisson
    ns.alpha = thermal_expansion
    # Final temperature
    ns.finalT = 0.0

    ns.δ = function.eye(domain.ndims)
    ns.X_i = 'x_i + u_i'
    ns.e_ij = '((∇_j(u_i) + ∇_i(u_j)) / 2) - alpha (T - finalT) δ_ij'
    ns.c_ijkl = 'lmbda δ_ij δ_kl + mu (δ_ik δ_jl + δ_il δ_jk)'
    ns.stress_ij = 'c_ijkl e_kl'

    # Constrain the inner radius on the bottom to restrict rigid body movement
    sqr = domain.boundary['bottom'].boundary['front'].integral('u_i u_i dL' @ ns, degree=2 * degree)
    ucons = solver.optimize('u,', sqr, droptol=1e-12, arguments=args)

    # Solve the momentum balance linearly
    ures = domain.integral('∇_j(v_i) stress_ij dV' @ ns, degree=2 * degree)

    args = solver.solve_linear('u:v',
                               ures,
                               constrain=ucons,
                               arguments=args)

    # Sample the solution and export VTK
    bezier = domain.boundary.sample('bezier', 5)
    x, X, initialT, stress, normU = bezier.eval(
        [ns.x, ns.X, ns.T, ns.stress, np.linalg.norm(ns.u)],
        **args)
    export.vtk('deformed_cylinder', bezier.tri, X, initialT=initialT, u=normU)

    # Export matplotlib
    with export.mplfigure('displacement.png') as fig:
        ax = fig.add_subplot(111, projection='3d')

        meanU = np.array([np.mean(normU[t]) for t in bezier.tri])
        norm = Normalize(np.min(meanU), np.max(meanU))
        surf = ax.plot_trisurf(X[:, 0], X[:, 1], X[:, 2], triangles=bezier.tri)
        surf.set_fc(coolwarm(norm(meanU)))

        cbar = fig.colorbar(ScalarMappable(cmap=coolwarm, norm=norm), ax=ax)
        cbar.set_label('Displacement')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')


if __name__ == '__main__':
    cli.run(main)
