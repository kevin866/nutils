from nutils import mesh, function, solver, export, testing
from nutils.expression_v2 import Namespace
import treelog
import numpy as np

def main(nelems=4,
         degree=3,
         stddev=0.05,
         nrefine=1,
         poisson=0.3,
         diffusivity=0.01,
         thermal_expansion=0.01,
         timestep=0.1,
         endtime=0.2,
         etype: str = 'square',
         btype: str = 'std'
         ):
    
    ns = Namespace()

    domain, ns.x0 = mesh.rectilinear([np.linspace(0,1,nelems+1)]*3)
    ns.define_for('x0', jacobians=('dV0',))


    basis = domain.basis('spline', degree=degree)
    ns.add_field('x', basis, shape=(domain.ndims,))
    ns.define_for('x', gradient='∇', jacobians=('dV', 'dS', 'dL'))
    ns.define_for('x', gradient='∇', normal='n', jacobians=('dV', 'dS'))
    ns.add_field(('u', 'v'), domain.basis(btype, degree=degree))

    args = solver.optimize('x,', domain.inegral('(x_i-x0_i) (x_i - x0_i) dV0' @ ns, degree = 2 * degree))

    rng = np.random.default_rng(seed=0)
    args['x'] = args['x'] + rng.normal(scale=stddev, size = args['x'].shape)

    

    return