from . import util, numpy, _

class ElemEval( object ):
  '''Element evaluation.

  Combines a specific Element instance with a shared LocalPoints
  instance, typically a set of integration points for this specific
  element type.
  
  Members:
   * elem: Element instance
   * points: LocalPoints instance
   * weights: integration weights (optional)'''

  def __init__( self, elem, points, transform=1 ):
    'constructor'

    assert isinstance( points, LocalPoints )
    assert points.ndims == elem.ndims
    self.elem = elem
    self.points = points
    self.weights = points.weights
    self.transform = transform

  def offset( self, offset ):
    'shift points'

    shifted = LocalPoints( self.points.coords + offset )
    return ElemEval( self.elem, shifted, self.transform )

  @util.cacheprop
  def next( self ):
    'get parent'

    if self.elem.parent is None:
      raise AttributeError, 'next'

    elem, transform = self.elem.parent
    points = transform.eval( self.points )
    return ElemEval( elem, points, transform.transform )

class Element( object ):
  '''Element base class.

  Represents the topological shape.'''

  def eval( self, where ):
    'get points'

    if isinstance( where, str ):
      points = self.getischeme( self.ndims, where )
    else:
      where = numpy.asarray( where )
      points = LocalPoints( where )
    return ElemEval( self, points )

  def zoom( self, elemset, points ):
    'zoom points'

    elem = self
    totaltransform = 1
    while elem not in elemset:
      elem, transform = self.parent
      points = transform( points )
      totaltransform = numpy.dot( transform.transform, totaltransform )
    return elem, points, totaltransform

class AffineTransformation( object ):
  'affine transformation'

  def __init__( self, offset, transform ):
    'constructor'

    self.offset = numpy.asarray( offset )
    self.transform = numpy.asarray( transform )

  @util.cachefunc
  def eval( self, points ):
    'apply transformation'

    if self.transform.ndim == 0:
      coords = self.offset[:,_] + self.transform * points.coords
    elif self.transform.shape[1] == 0:
      assert points.coords.shape == (0,1)
      coords = self.offset[:,_]
    else:
      coords = self.offset[:,_] + numpy.dot( self.transform, points.coords )
    return LocalPoints( coords, points.weights )

class CustomElement( Element ):
  'custom element'

  def __init__( self, **ischemes ):
    'constructor'

    self.ischemes = ischemes

  def getischeme( self, ndims, where ):
    'get integration scheme'

    assert ndims == self.ndims
    return self.ischemes[ where ]

class QuadElement( Element ):
  'quadrilateral element'

  def __init__( self, ndims, parent=None ):
    'constructor'

    self.ndims = ndims
    self.parent = parent
    Element.__init__( self )

  @util.classcache
  def edgetransform( cls, ndims ):
    'edge transforms'

    transforms = []
    for idim in range( ndims ):
      for iside in range( 2 ):
        offset = numpy.zeros( ndims )
        offset[idim:] = 1-iside
        offset[:idim+1] = iside
        transform = numpy.zeros(( ndims, ndims-1 ))
        transform.flat[ :(ndims-1)*idim :ndims] = 1 - 2 * iside
        transform.flat[ndims*(idim+1)-1::ndims] = 2 * iside - 1
        transforms.append( AffineTransformation( offset=offset, transform=transform ) )
    return transforms

  def edge( self, iedge ):
    'edge'

    transform = self.edgetransform( self.ndims )[ iedge ]
    return QuadElement( self.ndims-1, parent=(self,transform) )

  @util.classcache
  def refinedtransform( cls, ndims, n ):
    'refined transform'

    transforms = []
    transform = 1. / n
    for i in range( n**ndims ):
      offset = numpy.zeros( ndims )
      for idim in range( ndims ):
        offset[ ndims-1-idim ] = transform * ( i % n )
        i //= n
      transforms.append( AffineTransformation( offset=offset, transform=transform ) )
    return transforms

  @util.cachefunc
  def refined( self, n ):
    'refine'

    return [ QuadElement( self.ndims, parent=(self,transform) ) for transform in self.refinedtransform( self.ndims, n ) ]

  @util.classcache
  def getischeme( cls, ndims, where ):
    'get integration scheme'

    if ndims == 0:
      return LocalPoints( numpy.zeros([0,1]), numpy.array([1.]) )

    x = w = None
    if where.startswith( 'gauss' ):
      N = int( where[5:] )
      k = numpy.arange( 1, N )
      d = k / numpy.sqrt( 4*k**2-1 )
      x, w = numpy.linalg.eigh( numpy.diagflat(d,-1) ) # eigh operates (by default) on lower triangle
      w = w[0]**2
      x = ( x + 1 ) * .5
    elif where.startswith( 'uniform' ):
      N = int( where[7:] )
      x = numpy.arange( .5, N ) / N
      w = util.appendaxes( 1./N, N )
    elif where.startswith( 'bezier' ):
      N = int( where[6:] )
      x = numpy.linspace( 0, 1, N )
      w = util.appendaxes( 1./N, N )
    elif where.startswith( 'subdivision' ):
      N = int( where[11:] ) + 1
      x = numpy.linspace( 0, 1, N )
      w = None
    elif where.startswith( 'contour' ):
      N = int( where[7:] )
      p = numpy.linspace( 0, 1, N )
      if ndims == 1:
        coords = p[_]
      elif ndims == 2:
        coords = numpy.array([ p[ range(N) + [N-1]*(N-2) + range(N)[::-1] + [0]*(N-2) ],
                               p[ [0]*(N-1) + range(N) + [N-1]*(N-2) + range(1,N)[::-1] ] ])
      elif ndims == 3:
        assert N == 0
        coords = numpy.array([ [0,0,0], [1,0,0], [0,1,0], [1,1,0], [0,0,1], [1,0,1], [0,1,1], [1,1,1] ]).T
      else:
        raise Exception, 'contour not supported for ndims=%d' % ndims
    else:
      raise Exception, 'invalid element evaluation %r' % where
    if x is not None:
      coords = reduce( lambda coords, i:
        numpy.concatenate(( x[:,_].repeat( N**i, 1 ).reshape( 1, -1 ),
                       coords[:,_].repeat( N,    1 ).reshape( i, -1 ) )), range( 1, ndims ), x[_] )
    if w is not None:
      weights = reduce( lambda weights, i: ( weights * w[:,_] ).ravel(), range( 1, ndims ), w )
    else:
      weights = None
    return LocalPoints( coords, weights )

  def __repr__( self ):
    'string representation'

    return '%s#%x<ndims=%d>' % ( self.__class__.__name__, id(self), self.ndims )

class TriangularElement( Element ):
  'triangular element'

  ndims = 2
  edgetransform = (
    AffineTransformation( offset=[0,0], transform=[[ 1],[ 0]] ),
    AffineTransformation( offset=[1,0], transform=[[-1],[ 1]] ),
    AffineTransformation( offset=[0,1], transform=[[ 0],[-1]] ) )

  def __init__( self, parent=None ):
    'constructor'

    self.parent = parent
    Element.__init__( self )

  def edge( self, iedge ):
    'edge'

    transform = self.edgetransform[ iedge ]
    return QuadElement( ndims=1, parent=(self,transform) )

  @util.classcache
  def refinedtransform( cls, n ):
    'refined transform'

    transforms = []
    scale = 1./n
    for i in range( n ):
      transforms.extend( AffineTransformation( offset=numpy.array( [i,j], dtype=float ) / n, transform=scale ) for j in range(0,n-i) )
      transforms.extend( AffineTransformation( offset=numpy.array( [n-j,n-i], dtype=float ) / n, transform=-scale ) for j in range(n-i,n) )
    return transforms

  def refined( self, n ):
    'refine'

    if n == 1:
      return self
    return [ TriangularElement( parent=(self,transform) ) for transform in self.refinedtransform( n ) ]

  @util.classcache
  def getischeme( cls, ndims, where ):
    '''get integration scheme
    gaussian quadrature: http://www.cs.rpi.edu/~flaherje/pdf/fea6.pdf
    '''

    assert ndims == 2
    if where.startswith( 'contour' ):
      n = int( where[7:] or 0 )
      p = numpy.arange( n+1, dtype=float ) / (n+1)
      z = numpy.zeros_like( p )
      coords = numpy.hstack(( [1-p,p], [z,1-p], [p,z] ))
      weights = None
    elif where == 'gauss1':
      coords = numpy.array( [[1],[1]] ) / 3.
      weights = numpy.array( [1] ) / 2.
    elif where in 'gauss2':
      coords = numpy.array( [[4,1,1],[1,4,1]] ) / 6.
      weights = numpy.array( [1,1,1] ) / 6.
    elif where == 'gauss3':
      coords = numpy.array( [[5,9,3,3],[5,3,9,3]] ) / 15.
      weights = numpy.array( [-27,25,25,25] ) / 96.
    elif where == 'gauss4':
      A = 0.091576213509771; B = 0.445948490915965; W = 0.109951743655322
      coords = numpy.array( [[1-2*A,A,A,1-2*B,B,B],[A,1-2*A,A,B,1-2*B,B]] )
      weights = numpy.array( [W,W,W,1/3.-W,1/3.-W,1/3.-W] ) / 2.
    elif where == 'gauss5':
      A = 0.101286507323456; B = 0.470142064105115; V = 0.125939180544827; W = 0.132394152788506
      coords = numpy.array( [[1./3,1-2*A,A,A,1-2*B,B,B],[1./3,A,1-2*A,A,B,1-2*B,B]] )
      weights = numpy.array( [1-3*V-3*W,V,V,V,W,W,W] ) / 2.
    elif where == 'gauss6':
      A = 0.063089014491502; B = 0.249286745170910; C = 0.310352451033785; D = 0.053145049844816; V = 0.050844906370207; W = 0.116786275726379
      VW = 1/6. - (V+W) / 2.
      coords = numpy.array( [[1-2*A,A,A,1-2*B,B,B,1-C-D,1-C-D,C,C,D,D],[A,1-2*A,A,B,1-2*B,B,C,D,1-C-D,D,1-C-D,C]] )
      weights = numpy.array( [V,V,V,W,W,W,VW,VW,VW,VW,VW,VW] ) / 2.
    elif where == 'gauss7':
      A = 0.260345966079038; B = 0.065130102902216; C = 0.312865496004875; D = 0.048690315425316; U = 0.175615257433204; V = 0.053347235608839; W = 0.077113760890257
      coords = numpy.array( [[1./3,1-2*A,A,A,1-2*B,B,B,1-C-D,1-C-D,C,C,D,D],[1./3,A,1-2*A,A,B,1-2*B,B,C,D,1-C-D,D,1-C-D,C]] )
      weights = numpy.array( [1-3*U-3*V-6*W,U,U,U,V,V,V,W,W,W,W,W,W] ) / 2.
    elif where[:7] == 'uniform' or where[:6] == 'bezier':
      if where[:7] == 'uniform':
        N = int( where[7:] )
        points = ( numpy.arange( N ) + 1./3 ) / N
      else:
        N = int( where[6:] )
        points = numpy.linspace( 0, 1, N )
      NN = N**2
      C = numpy.empty( [2,N,N] )
      C[0] = points[:,_]
      C[1] = points[_,:]
      coords = C.reshape( 2, NN )
      flip = coords[0] + coords[1] > 1
      coords[:,flip] = 1 - coords[::-1,flip]
      weights = util.appendaxes( .5/NN, NN )
    else:
      raise Exception, 'invalid element evaluation: %r' % where
    return LocalPoints( coords, weights )

  def __repr__( self ):
    'string representation'

    return '%s#%x' % ( self.__class__.__name__, id(self) )

class LocalPoints( object ):
  'local point coordinates'

  def __init__( self, coords, weights=None ):
    'constructor'

    self.coords = coords
    self.weights = weights
    self.ndims, self.npoints = coords.shape

class StdElem( object ):
  'stdelem base class'

  pass

class PolyQuad( StdElem ):
  'poly quod'

  @util.classcache
  def __new__( cls, degree ):
    'constructor'

    self = object.__new__( cls )
    self.degree = degree
    self.ndims = len( degree )
    return self

  @util.classcache
  def comb( cls, p ):
    'comb'

    comb = numpy.empty( p, dtype=int )
    comb[0] = 1
    for j in range( 1, p ):
      comb[j] = ( comb[j-1] * (p-j) ) / j
    assert comb[-1] == 1
    return comb

  def __repr__( self ):
    'string representation'

    return 'PolyQuad#%x<degree=%s>' % ( id(self), ','.join( map( str, self.degree ) ) )

  @util.cachefunc
  def eval( self, points, grad=0 ):
    'evaluate'

    polydata = [ ( x, p, self.comb(p) ) for ( x, p ) in zip( points.coords, self.degree ) ]
    nshapes = numpy.prod( self.degree )

    F0 = [ numpy.array( [ [1.] ] if p == 1
                   else [ comb[i] * (1-x)**(p-1-i) * x**i for i in range(p) ]
                      ) for x, p, comb in polydata ]
    if grad == 0:
      return reduce( lambda f, fi: ( f[:,_] * fi ).reshape( -1, points.npoints ), F0 )

    F1 = [ numpy.array( [ [0.] ] if p < 2
                   else [ [-1.],[1.] ] if p == 2
                   else [ (1-p) * (1-x)**(p-2) ]
                      + [ comb[i] * (1-x)**(p-i-2) * x**(i-1) * (i-(p-1)*x) for i in range(1,p-1) ]
                      + [ (p-1) * x**(p-2) ]
                      ) for x, p, comb in polydata ]
    if grad == 1:
      data = numpy.empty(( nshapes, points.ndims, points.npoints ))
      for n in range( points.ndims ):
        Gi = [( F1 if m == n else F0 )[m] for m in range( points.ndims ) ]
        data[:,n] = reduce( lambda g, gi: ( g[:,_] * gi ).reshape( g.shape[0] * gi.shape[0], -1 ), Gi )
      return data

    F2 = [ numpy.array( [ [0.] ] * p if p < 3
                   else [ [2.],[-4.],[2.] ] if p == 3
                   else [ (p-1) * (p-2) * (1-x)**(p-3), (p-1) * (p-2) * (1-x)**(p-4) * ((p-1)*x-2) ]
                      + [ comb[i] * (1-x)**(p-i-3) * x**(i-2) * (x*(2*i-(p-1)*x)*(2-p)+i*(i-1)) for i in range(2,p-2) ]
                      + [ (p-1) * (p-2) * x**(p-4) * ((p-1)*(1-x)-2), (p-1) * (p-2) * x**(p-3) ]
                      ) for x, p, comb in polydata ]
    if grad == 2:
      data = numpy.empty(( nshapes, points.ndims, points.ndims, points.npoints ))
      for ni in range( points.ndims ):
        for nj in range( ni, points.ndims ):
          Di = [( F2 if m == ni == nj else F1 if m == ni or m == nj else F0 )[m] for m in range( points.ndims ) ]
          data[:,nj,ni] = data[:,ni,nj] = reduce( lambda d, di: ( d[:,_] * di ).reshape( d.shape[0] * di.shape[0], -1 ), Di )
      return data

    F3 = [ numpy.array( [ [0.] ] * p if p < 4
                   else [ [-6],[18],[-18],[6] ] if p == 4
                   else [ 24*(x-1), 72-96*x, 72*(2*x-1), 24-96*x, 24*x ] if p == 5
                   else [ -(p-3)*(p-2)*(p-1)*(1-x)**(p-4), -(-6+11*p-6*p**2+p**3)*(1-x)**(p-5)*((p-1)*x-3), -comb[2]*(p-3)*(1-x)**(p-6)*(6-6*(p-2)*x+(2-3*p+p**2)*x**2) ]
                      + [ comb[i]*(1-x)**(p-i-4)*x**(i-3)*(i**3-(-6+11*p-6*p**2+p**3)*x**3-3*i**2*(1+(p-3)*x)+i*(2+3*(p-3)*x+3*(6-5*p+p**2)*x**2)) for i in range(3,p-3) ]
                      + [ comb[p-3]*(p-3)*x**(p-6)*(p**2*(x-1)**2+2*(10-8*x+x**2)-3*p*(3-4*x+x**2)), -(-6+11*p-6*p**2+p**3)*(4+p*(x-1)-x)*x**(p-5), (p-3)*(p-2)*(p-1)*x**(p-4) ]
                      ) for x, p, comb in polydata ]

    if grad == 3:
      data = numpy.empty(( nshapes, points.ndims, points.ndims, points.ndims, points.npoints ))
      for ni in range( points.ndims ):
        for nj in range( ni, points.ndims ):
          for nk in range( nj, points.ndims ):
            Dijk = [( F3 if m == ni == nj == nk
                 else F2 if m == ni == nj or m == ni == nk or m == nj == nk
                 else F1 if m == ni or m == nj or m == nk
                 else F0 )[m] for m in range( points.ndims )]
            data[:,ni,nj,nk] = data[:,ni,nk,nj] = data[:,nj,ni,nk] = \
            data[:,nj,nk,ni] = data[:,nk,ni,nj] = data[:,nk,nj,ni] = \
              reduce( lambda d, di: ( d[:,_] * di ).reshape( d.shape[0] * di.shape[0], -1 ), Dijk )
      return data

    F4 = [ numpy.array( [ [0.] ] * p if p < 5
                   else [ [24], [-96], [24*comb[2]], [-96], [24] ] if p == 5
                   else [ -120*(x-1), 120*(5*x-4), 24*comb[2]*(3-5*x), 24*comb[p-3]*(5*x-2), 120-600*x, 120*x ] if p == 6
                   else [ 360*(x-1)**2, -720*(2-5*x+3*x**2), 24*comb[2]*(6-20*x+15*x**2), -72*comb[3]*(1-5*x+5*x**2), -72*comb[p-4]*(1-5*x+5*x**2), 24*comb[p-3]*(1-10*x+15*x**2), -720*x*(3*x-1), 360*x**2 ] if p == 7
                   else [ (p-4)*(p-3)*(p-2)*(p-1)*(1-x)**(p-5), (24-50*p+35*p**2-10*p**3+p**4)*(1-x)**(p-6)*((p-1)*x-4), comb[2]*(12-7*p+p**2)*(1-x)**(p-7)*(12-8*(p-2)*x+(2-3*p+p**2)*x**2),
                          comb[3]*(p-4)*(1-x)**(p-8)*(-24+36*(p-3)*x-12*(6-5*p+p**2)*x**2+(-6+11*p-6*p**2+p**3)*x**3) ]
                      + [ comb[i]*(1-x)**(p-i-5)*x**(i-4)*(i**4+(24-50*p+35*p**2-10*p**3+p**4)*x**4-2*i**3*(3+2*(p-4)*x)+i**2*(11+12*(p-4)*x+6*(12-7*p+p**2)*x**2)-2*i*(3+4*(p-4)*x+3*(12-7*p+p**2)*x**2 + 
                          2*(-24+26*p-9*p**2+p**3)*x**3)) for i in range(4,p-4) ]
                      + [ -comb[p-4]*(p-4)*x**(p-8)*(-6*p**2*(x-3)*(x-1)**2+p**3*(x-1)**3-6*(-35+45*x-15*x**2+x**3)+p*(-107+189*x-93*x**2+11*x**3)), comb[p-3]*(12-7*p+p**2)*x**(p-7)*(p**2*(x-1)**2 +
                          p*(-11+14*x-3*x**2)+2*(15-10*x+x**2)), -(24-50*p+35*p**2-10*p**3+p**4)*(5+p*(x-1)-x)*x**(p-6), (p-4)*(p-3)*(p-2)*(p-1)*x**(p-5) ]
                      ) for x, p, comb in polydata ]

    if grad == 4:
      data = numpy.empty(( nshapes, points.ndims, points.ndims, points.ndims, points.ndims, points.npoints ))
      for ni in range( points.ndims ):
        for nj in range( ni, points.ndims ):
          for nk in range( nj, points.ndims ):
            for nl in range( nk, points.ndims ):
              Dijkl = [( F4 if m == ni == nj == nk == nl
                    else F3 if m == ni == nj == nk or m == ni == nj == nl or m == ni == nk == nl or m == nj == nk == nl
                    else F2 if m == ni == nj or m == ni == nk or m == ni == nl or m == nj == nk or m == nj == nl or m == nk == nl
                    else F1 if m == ni or m == nj or m == nk or m == nl
                    else F0 )[m] for m in range( points.ndims )]
              data[:,ni,nj,nk,nl] = data[:,ni,nj,nl,nk] = data[:,ni,nk,nj,nl] = data[:,ni,nk,nl,nj] = data[:,ni,nl,nj,nk] = data[:,ni,nl,nk,nj] = \
              data[:,nj,ni,nk,nl] = data[:,nj,ni,nl,nk] = data[:,nj,nk,ni,nl] = data[:,nj,nk,nl,ni] = data[:,nj,nl,ni,nk] = data[:,nj,nl,nk,ni] = \
              data[:,nk,ni,nj,nl] = data[:,nk,ni,nl,nj] = data[:,nk,nj,ni,nl] = data[:,nk,nj,nl,ni] = data[:,nk,nl,ni,nj] = data[:,nk,nl,nj,ni] = \
              data[:,nl,ni,nj,nk] = data[:,nl,ni,nk,nl] = data[:,nl,nj,ni,nk] = data[:,nl,nj,nk,ni] = data[:,nl,nk,ni,nj] = data[:,nl,nk,nj,ni] = \
                reduce( lambda d, di: ( d[:,_] * di ).reshape( d.shape[0] * di.shape[0], -1 ), Dijkl )
      return data

    assert all( p <= 5 for p in self.degree ) # for now
    return numpy.zeros(( nshapes, points.ndims, points.ndims, points.ndims, points.npoints ))

class PolyTriangle( StdElem ):
  'poly triangle'

  @util.classcache
  def __new__( cls, order ):
    'constructor'

    assert order == 1
    self = object.__new__( cls )
    return self

  @util.cachefunc
  def eval( self, points, grad=0 ):
    'eval'

    if grad == 0:
      x, y = points.coords
      data = numpy.array( [ x, y, 1-x-y ] )
    elif grad == 1:
      data = numpy.array( [[[1],[0]],[[0],[1]],[[-1],[-1]]], dtype=float )
    else:
      data = numpy.array( 0 ).reshape( (1,) * (grad+1+points.ndim) )
    return data

  def __repr__( self ):
    'string representation'

    return '%s#%x' % ( self.__class__.__name__, id(self) )

class ExtractionWrapper( object ):
  'extraction wrapper'

  def __init__( self, stdelem, extraction ):
    'constructor'

    self.stdelem = stdelem
    self.extraction = extraction

  @util.cachefunc
  def eval( self, points, grad=0 ):
    'call'

    return numpy.dot( self.stdelem.eval( points, grad ).T, self.extraction.T ).T

  def __repr__( self ):
    'string representation'

    return '%s#%x:%s' % ( self.__class__.__name__, id(self), self.stdelem )

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
