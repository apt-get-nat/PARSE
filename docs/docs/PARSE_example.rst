.. code:: ipython3

    import numpy as np
    import parse
    import pyvista as pv
    import importlib

.. code:: ipython3

    importlib.reload(parse)
    parse.pullPARSE(7821,1)
    data = parse.readPARSE((7821,1))

.. code:: ipython3

    data.compile_grid()
    xseed,yseed = np.meshgrid(np.linspace(0,max(data.nodes[:,0]),20),np.linspace(0,max(data.nodes[:,1]),20))
    seedpoints = np.vstack((xseed.ravel(),yseed.ravel(),np.zeros(20**2))).transpose()
    fieldlines = data.fieldlines(seedpoints)

.. code:: ipython3

    pl = pv.Plotter()
    Zmesh = pv.ImageData()
    Zmesh.dimensions = np.array([len(data.Bgrid.xcoords),len(data.Bgrid.ycoords),len(data.Bgrid.zcoords)])+1
    Zmesh.spacing = data.Bgrid.grid_spacing
    Zmesh.cell_data["values"] = data.Bgrid.vectors[:,:,:,2].ravel("F")
    pl.add_mesh(Zmesh.slice(normal=[0,0,1],origin=[1e-2,1e-2,1e-2]))
    for stream in fieldlines:
        pl.add_lines(stream,color='black',width=1,connected = True)
    pl.show(jupyter_backend='trame')



.. parsed-literal::

    Widget(value="<iframe src='http://localhost:58198/index.html?ui=P_0x26c066a75e0_0&reconnect=auto' style='width…


