import numpy as np

fformatstr = "{folder}/PARSE.simulation.1.0.0.sharp.{sharpnum}.realization.{parseind}.fits"
getrecord = 8213061

def pullPARSE(sharpnum, parseind=0, folder="./parsedata", overwrite=False):
    """
    Download a specified PARSE data element.

    :param sharpnum: Integer corresponding to the SHARP number the PARSE was generated from.
    :type kind: int
    :param parseind: The realization index for the desired PARSE. Omit or set to 0 for the NLFFF extrapolation
    :type parseind: int
    :param folder: Optional specification of the directory the data should be downloaded to.
    :type folder: String
    :param overwrite: Whether new data should be re-downloaded in the event a PARSE with the specified sharp and
                      parse indices is present in the download folder. Default False
    :type overwrite: bool
    """
    import urllib.request
    from os import path, mkdir
    
    if not path.exists(fformatstr.format(folder=folder,sharpnum=sharpnum,parseind=parseind)):
        if not path.exists(folder):
            mkdir(folder)
        urllib.request.urlretrieve(f"https://zenodo.org/record/{getrecord}/files/PARSE.simulation.1.0.0.sharp.{sharpnum}.realization.{parseind}.fits?download=1", fformatstr.format(folder=folder,sharpnum=sharpnum,parseind=parseind))


def readPARSE(fid,folder="./parsedata"):
    """
    Read a specified PARSE fits file in as a :py:class:`PARSE` object.
    
    :param fid: A specification of the file to read. If ``fid`` is a ``String``,
                it will read that file. Include the relative directory; the ``folder``
                parameter is ignored. If ``fid`` is a tuple ``(sharpnum, parseind)``,
                it will look for a file with that specification.
    :type fid: String or tuple
    :param folder: the folder in which to find the fits file, if the tuple specification of
                   ``fid`` is used.
    :type folder: String
    
    :return: The data object read in from the file
    :rtype: :py:class:`PARSE`
    """
    from astropy.io import fits
    
    if type(fid) is tuple:
        fname=fformatstr.format(folder=folder,sharpnum=fid[0],parseind=fid[1])
    elif isinstance(fid,str):
        fname=fid
    else:
        raise TypeError("fid must be a string or tuple")
    hdul = fits.open(fname)
    return PARSE(hdul)

class PARSE:
    """
    A class that holds and manages the scattered vector magnetic field data from a PARSE simulation.
    
    :param hdul: An astropy-opened FITS header list from the current PARSE version.
    :type hdul: :py:class:`astropy.io.fits.HDUList`
    
    :ivar B: n x 3 :py:class:`numpy.ndarray` containing the Bx, By, Bz vector components at each defined point in space.
    :ivar nodes: n x 3 :py:class:`numpy.ndarray` containing the x, y and z coordinates of each defined point in space.
    :ivar F: n x 3 :py:class:`numpy.ndarray` containing the plasma forcing vector Fx, Fy and Fz components at each defined
             point in space.
    :ivar header: a ``dict`` containing the metadata for the PARSE run, including its spatial size and coordinates at time
                  of observation. For a full list of keys, see `Mathews & Thompson 2023 <https://arxiv.org/abs/2308.02138>`_,
                  Table 2.
    """
    def __init__(self,hdul):
        """
        Constructor method
        """
        self.B = np.vstack((hdul[1].data,hdul[2].data,hdul[3].data)).transpose()
        self.nodes = np.vstack((hdul[4].data,hdul[5].data,hdul[6].data)).transpose()
        self.F = np.vstack((hdul[7].data,hdul[8].data,hdul[9].data)).transpose()
        
        self._lenX = np.max(self.nodes[:,0])
        self._lenY = np.max(self.nodes[:,1])
        self._lenZ = np.max(self.nodes[:,2])
        self.header = {key:hdul[0].header[key] for key in hdul[0].header.keys()}
        
    def compile_grid(self,res=100):
        """
        Adds an instance variable which can be used to plot. :py:class:`scipy.interpolate.NearestNDInterpolator`
        is used to perform the interpolation.
        
        :param res: The minimum number of points in each dimension of the interpolated grid.
                    Defaults to 100. In the event field lines are traced, res also determines the step size and
                    step maximum for the streamline tracer.
        :type res: int
        
        :ivar Bgrid: :py:class:`streamtracer.VectorGrid` object defining the interpolated grid.
        """
        import streamtracer
        from scipy import interpolate
        
        step = np.min([self._lenX,self._lenY,self._lenZ])/res
        self._stracer = streamtracer.StreamTracer(res*100,step)
        # Note this interpolator could be improved to a high-order rbf one easily;
        # such an interpolator is also in scipy.interpolate. But it is much more
        # costly at preconstruction time.
        Bxinterp = interpolate.NearestNDInterpolator(self.nodes,self.B[:,0])
        Byinterp = interpolate.NearestNDInterpolator(self.nodes,self.B[:,1])
        Bzinterp = interpolate.NearestNDInterpolator(self.nodes,self.B[:,2])
        points = (np.arange(0,self._lenX,step),np.arange(0,self._lenY,step),np.arange(0,self._lenZ,step))
        xgrid,ygrid,zgrid = np.meshgrid(points[0],points[1],points[2],indexing='ij')
        Bxgrid = Bxinterp(xgrid.ravel(),ygrid.ravel(),zgrid.ravel())
        Bygrid = Byinterp(xgrid.ravel(),ygrid.ravel(),zgrid.ravel())
        Bzgrid = Bzinterp(xgrid.ravel(),ygrid.ravel(),zgrid.ravel())
        Bgrid = np.vstack((Bxgrid,Bygrid,Bzgrid)).transpose()
        Bgrid = Bgrid.reshape((len(points[0]),len(points[1]),len(points[2]),3))
        self.Bgrid = streamtracer.VectorGrid(Bgrid, grid_coords=points)
        self.Bgrid.grid_spacing = np.array([step,step,step])
    def fieldlines(self,seedpoints):
        """
        Wrapper for :py:func:`streamtracer.StreamTracer.trace`. Returns the fieldlines, and also stores them in ``_stracer.xs``.
        
        :param seedpoints: m x 3 array containing the x,y,z coordinates for the starting point of the field lines
        :type seedpoints: :py:class:`numpy.ndarray`
        
        :return: A list of coordinates for each field line.
        :rtype: list[:py:class:`numpy.ndarray`]
        """
        if 'Bgrid' not in dir(self):
            raise AttributeError('fieldlines expects compile_grid to have been called first, but no Bgrid attribute found.')
        self._stracer.trace(seedpoints,self.Bgrid)
        return self._stracer.xs
