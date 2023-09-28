.. _basic usage:

Basic Usage
***********

First we must import the library:
``
import readparse as parse
``

To download a single element from the PARSE dataset (say, corresponding to SHARP 7821), use the ``pullPARSE`` function. By default, this downloads it to a local directory.

``
parse.pullPARSE(7821,0)
``

Read the data with the ``readPARSE`` function, which takes a tuple of the same arguments pullPARSE got.

``
data = parse.readPARSE((7821,0))
``

Now data is a ``PARSE`` object, containing the full scattered data, which can be worked with as-is. Suppose we wish to visualize this element. It is easiest to do this if we first interpolate onto a regular grid, so let us do so.

``
data.compile_grid()
``

Now we'll trace some fieldlines for plotting.

``
import numpy as np

xseed,yseed = np.meshgrid(np.linspace(0,max(data.nodes[:,0]),20),np.linspace(0,max(data.nodes[:,1]),20))
seedpoints = np.vstack((xseed.ravel(),yseed.ravel(),np.zeros(20**2))).transpose()
fieldlines = data.fieldlines(seedpoints)
```

Finally, we plot the fieldlines and lower boundary in vtk.
``
import pyvista as pv

pl = pv.Plotter()
Zmesh = pv.ImageData()
Zmesh.dimensions = np.array([len(data.Bgrid.xcoords),len(data.Bgrid.ycoords),len(data.Bgrid.zcoords)])+1
Zmesh.spacing = data.Bgrid.grid_spacing
Zmesh.cell_data["values"] = data.Bgrid.vectors[:,:,:,2].ravel("F")
pl.add_mesh(Zmesh.slice(normal=[0,0,1],origin=[1e-2,1e-2,1e-2]))
for stream in fieldlines:
    pl.add_lines(stream,color='black',width=1,connected = True)
pl.show()
``