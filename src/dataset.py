
import os
import logging
import numpy as np
import skimage.color
import skimage.io
import skimage.transform
import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool
from astropy.io.fits import getdata
from astropy.io import fits


class Dataset(object):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:

    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...

    See COCODataset and ShapesDataset as examples.
    """

    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id):
        """Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.

        Override for your dataset, but pass to this function
        if you encounter images not in your dataset.
        """
        return ""

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.

        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        # Mapping from source class and image IDs to internal IDs
        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.image_info, self.image_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.

        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's available online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances].

        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        # Override this function to load a mask from your dataset.
        # Otherwise, it returns an empty mask.
        logging.warning("You are using the default load_mask(), maybe you need to define your own one.")
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids

class PhoSimDataset(Dataset):

    def __init__(self, height=512, width=512, stretch=0.005, Q=10, m=0):
        self.height = height
        self.width = width
        # Contrast scaling parameters
        self.stretch = stretch
        self.Q = Q
        self.m = m
        super(PhoSimDataset, self).__init__()


    def load_sources(self, set_dir, dataset="validation", normalize="zscore", store_raw=False):
        # Load sources in dataset with proper id
        # This happens once, upon calling dataset.prepare()
        self.dataset = dataset
        self.out_dir = set_dir
        # load specifications for image Dataset
        # follows load_shapes example
        black = (0,0,0)
        # add DES classes
        self.add_class("des", 1, "star")
        self.add_class("des", 2, "galaxy")

        # find number of sets:
        num_sets = 0
        for setdir in os.listdir(self.out_dir):
            if 'set_' in setdir:
                # add tranining image set
                self.add_image("des", image_id=num_sets,path=os.path.join(self.out_dir,set_dir),
                    width=self.width,height=self.height,bg_color=black)
                num_sets += 1

        # store data in memory
        self.images = [None]*(num_sets)
        if store_raw:
            self.raws = [None]*(num_sets)

        self.masks = [None]*num_sets
        self.class_ids_mem = [None]*num_sets
        threads = np.clip(mp.cpu_count(),1,num_sets)
        print("Loading images from disk.")
        pool = ThreadPool(threads)
        pool.starmap(self.load_image_disk, [(i, normalize, store_raw) for i in range(num_sets)])
        if dataset == "training" or dataset == "validation":
            print("Loading masks from disk (this may take several minutes).")
            pool.map(self.load_mask_disk, range(num_sets))
        pool.close()
        pool.join()
        return

    def load_image(self, image_id, raw = False):
        if raw:
            return self.raws[image_id]
        else:
            return self.images[image_id]

    def load_image_disk(self, image_id, normalize='zscore', store_raw = False):
        # load from disk -- each set directory contains seperate files for images and masks
        info = self.image_info[image_id]
        setdir = 'set_%d' % image_id
        # read images
        g = getdata(os.path.join(self.out_dir,setdir,"img_g.fits"),memmap=False)
        r = getdata(os.path.join(self.out_dir,setdir,"img_r.fits"),memmap=False)
        z = getdata(os.path.join(self.out_dir,setdir,"img_z.fits"),memmap=False)

        image = np.zeros([info['height'], info['width'], 3], dtype=np.int16)

        # store raw image
        if store_raw:
            image_raw = np.zeros([info['height'], info['width'], 3], dtype=np.float64)
            
            image_raw[:,:,0] = z # red
            image_raw[:,:,1] = r # green
            image_raw[:,:,2] = g # blue
            self.raws[image_id] = image_raw

        # Contrast scaling / normalization
        I = (z+r+g)/3.0
        stretch = self.stretch
        Q = self.Q
        m = self.m

        if normalize == 'lupton':
            z = z*np.arcsinh(stretch*Q*(I - m))/(Q*I)
            r = r*np.arcsinh(stretch*Q*(I - m))/(Q*I)
            g = g*np.arcsinh(stretch*Q*(I - m))/(Q*I)
        elif normalize == 'zscore':
            Isigma = I*np.mean([np.std(g),np.std(r),np.std(z)])
            z = (z - np.mean(z) - m)/Isigma
            r = (r - np.mean(r) - m)/Isigma
            g = (g - np.mean(g) - m)/Isigma
        elif normalize == 'linear':
            z = (z - m)/I
            r = (r - m)/I
            g = (g - m)/I

        max_RGB = np.percentile([z,r,g], 99.995)
        # avoid saturation
        r = r/max_RGB; g = g/max_RGB; z = z/max_RGB

        # Rescale to 16-bit int
        int16_max = np.iinfo(np.int16).max
        r = r * int16_max
        g = g * int16_max
        z = z * int16_max

        image[:,:,0] = z # red
        image[:,:,1] = r # green
        image[:,:,2] = g # blue
        self.images[image_id] = image
        return image

    def load_mask(self, image_id):
        return self.masks[image_id], self.class_ids_mem[image_id]

    def load_mask_disk(self, image_id):
        # Load from disk
        info = self.image_info[image_id]
        # load image set via image_id from phosim output directory
        setdir = 'set_%d' % image_id
        maskdir = os.path.join(self.out_dir,setdir,"masks.fits")
        with fits.open(maskdir,memmap=False,lazy_load_hdus=False) as hdul:
            sources = len(hdul)
            data = [hdu.data/np.max(hdu.data) for hdu in hdul]
            class_ids = [hdu.header["CLASS_ID"] for hdu in hdul]
        # make mask from threshold
        thresh = [0.005 if i == 1 else 0.08 for i in class_ids]
        masks = np.zeros([info['height'], info['width'], sources],dtype=np.uint8)
        for i in range(sources):
            """
            # inital guess
            x0, y0 = np.unravel_index(np.argmax(data[i]), masks.shape)
            sma = 10 # semi-major axis
            eps = 0 # ellipticity
            g = EllipseGeometry(x0, y0, sma, eps, pa)
            ellipse = Ellipse(data, geometry=g)
            isolist = ellipse.fit_image()
            # convert Petrosian isophot to mask
            position = [isolist.x0, isolist.y0]
            sma = isolist.sma
            b = sma*np.sqrt(1-isolist.eps**2)
            aper = EllipticalAperture(position, sma, b, isolist.pa)
            # create mask
            masks[:,:,i] = aper.to_mask(method='subpixel')
            """
            masks[:,:,i][data[i]>thresh[i]] = 1
            masks[:,:,i] = cv2.GaussianBlur(masks[:,:,i],(9,9),2)
        self.class_ids_mem[image_id] = np.array(class_ids,dtype=np.uint8)
        self.masks[image_id] = np.array(masks,dtype=bool)
        return self.masks[image_id], self.class_ids_mem[image_id]