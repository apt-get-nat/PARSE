import numpy as np

__version__ = "1.0.0"

fformatstr = "{folder}/PARSE.simulation.1.0.0.sharp.{sharpnum}.realization.{parseind}.fits"
getrecord = 8213061

def pullPARSE(sharpnum, parseind=0, folder="./parsedata", overwrite=False):
    import urllib.request
    from os import path, mkdir
    
    if not path.exists(fformatstr.format(folder=folder,sharpnum=sharpnum,parseind=parseind)):
        if not path.exists(folder):
            mkdir(folder)
        urllib.request.urlretrieve(f"https://zenodo.org/record/{getrecord}/files/PARSE.simulation.1.0.0.sharp.{sharpnum}.realization.{parseind}.fits?download=1", fformatstr.format(folder=folder,sharpnum=sharpnum,parseind=parseind))


def readPARSE(fid,folder="./parsedata"):
    """
    if fid is a STRING, will read that file (must include directory)
    if fid is a tuple (sharpnum, parseind), will look for that file by standard name in provided folder.
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
    def __init__(self,hdul):
        """
        Assumes hdul is an astropy-opened header list from the current PARSE version. Silent errors and
        unstable behavior may occur otherwise.
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
        Provides a Bgrid key which can be used to plot.
        """
        import streamtracer
        from scipy import interpolate
        
        step = np.min([self._lenX,self._lenY,self._lenZ])/res
        self.stracer = streamtracer.StreamTracer(res*100,step)
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
        self.stracer.trace(seedpoints,self.Bgrid)
        return self.stracer.xs
