# Copyright (c) 2014 Evalf
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
The core module provides a collection of low level constructs that have no
dependencies on other nutils modules. Primarily for internal use.
"""

import sys, functools, os

globalproperties = {
  'nprocs': 1,
  'outrootdir': '~/public_html',
  'outdir': '',
  'verbose': 4,
  'richoutput': sys.stdout.isatty(),
  'htmloutput': True,
  'pdb': False,
  'imagetype': 'png',
  'symlink': '',
  'recache': False,
  'dot': False,
  'profile': False,
  'selfcheck': False,
}

if os.access( '/run/shm', os.W_OK ):
  globalproperties['shmdir'] = '/run/shm'

for nutilsrc in ['~/.config/nutils/config', '~/.nutilsrc']:
  nutilsrc = os.path.expanduser( nutilsrc )
  if not os.path.isfile( nutilsrc ):
    continue
  try:
    with open(nutilsrc) as rc:
      exec( rc.read(), {}, globalproperties )
  except:
    exc_value, frames = sys.exc_info()
    exc_str = '\n'.join( [ repr(exc_value) ] + [ str(f) for f in frames ] )
    print( 'Skipping .nutilsrc: {}'.format(exc_str) )
  break

_nodefault = object()
def getprop( name, default=_nodefault, frame=None ):
  """Access a semi-global property.

  The use of global variables is discouraged, as changes can go unnoticed and
  lead to abscure failure. The getprop mechanism makes local variables accesible
  (read only) from nested scopes, but not from the encompassing scope.

  >>> def f():
  ...   print(getprop('myval'))
  ...
  >>> def main():
  ...   __myval__ = 2
  ...   f()

  Args:
      name (str): Property name, corresponds to __name__ local variable.
      default: Optional default value.

  Returns:
      The object corresponding to the first __name__ encountered in a higher
      scope. If none found, return default. If no default specified, raise
      NameError.
  """

  key = '__%s__' % name
  if frame is None:
    frame = sys._getframe(1)
  while frame:
    if key in frame.f_locals:
      return frame.f_locals[key]
    frame = frame.f_back
  if name in globalproperties:
    return globalproperties[name]
  if default is _nodefault:
    raise NameError( 'property %r is not defined' % name )
  return default

def index( items ):
  """Index of the first nonzero item.

  Args:
      items: Any iterable object

  Returns:
      The index of the first item for which bool(item) returns True.
  """

  for i, item in enumerate(items):
    if item:
      return i
  raise ValueError

def single_or_multiple( f ):
  """
  Method wrapper, converts first positional argument to tuple: tuples/lists
  are passed on as tuples, other objects are turned into tuple singleton.
  Return values should match the length of the argument list, and are unpacked
  if the original argument was not a tuple/list.

  >>> class Test:
  ...   @single_or_multiple
  ...   def square(self, args):
  ...     return [v**2 for v in args]
  ...
  >>> T = Test()
  >>> T.square(2)
  4
  >>> T.square([2,3])
  [4, 9]

  Args:
      f: Method that expects a tuple as first positional argument, and that
      returns a list/tuple of the same length.

  Returns:
      Wrapped method.
  """

  @functools.wraps( f )
  def wrapped( self, arg0, *args, **kwargs ):
    ismultiple = isinstance( arg0, (list,tuple) )
    arg0mod = tuple(arg0) if ismultiple else (arg0,)
    retvals = f( self, arg0mod, *args, **kwargs )
    if not ismultiple:
      retvals, = retvals
    return retvals
  return wrapped

def open_in_outdir( file, *args, **kwargs ):
  '''open a file relative to the ``outdirfd`` or ``outdir`` property

  Wrapper around :func:`open` that opens a file relative to either the
  ``outdirfd`` property (if supported, see :func:`os.supports_dir_fd`) or
  ``outdir``.  Takes the same arguments as :func:`open`.
  '''

  assert 'opener' not in kwargs
  outdirfd = getprop( 'outdirfd', None )
  outdir = getprop( 'outdir', None )
  if outdirfd is not None and os.open in os.supports_dir_fd:
    kwargs['opener'] = functools.partial( os.open, dir_fd=outdirfd )
  elif outdir:
    file = os.path.join(os.path.expanduser(outdir), file)
  return open( file, *args, **kwargs )

def listoutdir():
  '''list files in ``outdirfd`` or ``outdir`` property'''

  outdirfd = getprop( 'outdirfd', None )
  outdir = getprop( 'outdir', None )
  if outdirfd is not None:
    return os.listdir( outdirfd )
  elif outdir:
    return os.listdir(os.path.expanduser(outdir))
  else:
    return os.listdir()


# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
