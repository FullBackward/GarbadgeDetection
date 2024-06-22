"""

Author: Pedro F. Proenza

"""

import os
import json
import numpy as np
import copy
import skimage
import utilsTACO

from PIL import Image, ExifTags

from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

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

    def append_data(self, class_info, image_info):
        self.external_to_class_id = {}
        for i, c in enumerate(self.class_info):
            for ds, id in c["map"]:
                self.external_to_class_id[ds + str(id)] = i

        # Map external image IDs to internal ones.
        self.external_to_image_id = {}
        for i, info in enumerate(self.image_info):
            self.external_to_image_id[info["ds"] + str(info["id"])] = i

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's availble online for easy
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
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids

class Taco(Dataset):
    def __init__(self, class_map=None):
        super().__init__()

    def load_taco(self, dataset_dir, round, subset, class_ids=None,
                  class_map=None, return_taco=False, auto_download=False):
        """Load a subset of the TACO dataset.
        dataset_dir: The root directory of the TACO dataset.
        round: split number
        subset: which subset to load (train, val, test)
        class_ids: If provided, only loads images that have the given classes.
        class_map: Dictionary used to assign original classes to new class system
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """

        # TODO: Once we got the server running
        # if auto_download is True:
        #     self.auto_download(dataset_dir, subset, year)
        ann_filepath = os.path.join(dataset_dir , 'annotations')
        if round != None:
            ann_filepath += "_" + str(round) + "_" + subset + ".json"
        else:
            ann_filepath += ".json"

        assert os.path.isfile(ann_filepath)

        # Load dataset
        dataset = json.load(open(ann_filepath, 'r'))

        # Replace dataset original classes before calling the coco Constructor
        # Some classes may be assigned background to remove them from the dataset
        self.replace_dataset_classes(dataset, class_map)

        taco_alla_coco = COCO()
        taco_alla_coco.dataset = dataset
        taco_alla_coco.createIndex()

        # Add images and classes except Background
        # Definitely not the most efficient way
        image_ids = []
        background_id = -1
        class_ids = sorted(taco_alla_coco.getCatIds())
        for i in class_ids:
            class_name = taco_alla_coco.loadCats(i)[0]["name"]
            if class_name != 'Background':
                self.add_class("taco", i, class_name)
                image_ids.extend(list(taco_alla_coco.getImgIds(catIds=i)))
            else:
                background_id = i
        image_ids = list(set(image_ids))

        if background_id > -1:
            class_ids.remove(background_id)

        print('Number of images used:', len(image_ids))

        # Add images
        for i in image_ids:
            self.add_image(
                "taco", image_id=i,
                path=os.path.join(dataset_dir, taco_alla_coco.imgs[i]['file_name']),
                width=taco_alla_coco.imgs[i]["width"],
                height=taco_alla_coco.imgs[i]["height"],
                annotations=taco_alla_coco.loadAnns(taco_alla_coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_taco:
            return taco_alla_coco

    def add_transplanted_dataset(self, dataset_dir, class_map = None):

        # Load dataset
        ann_filepath = os.path.join(dataset_dir, 'annotations.json')
        dataset = json.load(open(ann_filepath, 'r'))

        # Map dataset classes
        self.replace_dataset_classes(dataset, class_map)

        taco_alla_coco = COCO()
        taco_alla_coco.dataset = dataset
        taco_alla_coco.createIndex()

        class_ids = sorted(taco_alla_coco.getCatIds())

        # Select images by class
        # Add images
        image_ids = []
        background_id = -1
        for i in class_ids:
            class_name = taco_alla_coco.loadCats(i)[0]["name"]
            if class_name != 'Background':
                image_ids.extend(list(taco_alla_coco.getImgIds(catIds=i)))
                # TODO: Select how many
            else:
                background_id = i
        image_ids = list(set(image_ids))

        if background_id > -1:
            class_ids.remove(background_id)

        # Retrieve list of training image ids
        train_image_ids = [x['id'] for x in self.image_info]

        nr_train_images_so_far = len(train_image_ids)

        # Add images
        transplant_counter = 0
        for i in image_ids:
            if taco_alla_coco.imgs[i]['source_id'] in train_image_ids:
                transplant_counter += 1
                self.add_image(
                    "taco", image_id=i+nr_train_images_so_far,
                    path=os.path.join(dataset_dir, taco_alla_coco.imgs[i]['file_name']),
                    width=taco_alla_coco.imgs[i]["width"],
                    height=taco_alla_coco.imgs[i]["height"],
                    annotations=taco_alla_coco.loadAnns(taco_alla_coco.getAnnIds(
                        imgIds=[i], catIds=class_ids, iscrowd=None)))

        print('Number of transplanted images added: ', transplant_counter, '/', len(image_ids))

    def load_image(self, image_id):
        """Load the specified image and return as a [H,W,3] Numpy array."""

        # Load image. TODO: do this with opencv to avoid need to correct orientation
        image = Image.open(self.image_info[image_id]['path'])
        img_shape = np.shape(image)

        # load metadata
        exif = image._getexif()
        if exif:
            exif = dict(exif.items())
            # Rotate portrait images if necessary (274 is the orientation tag code)
            if 274 in exif:
                if exif[274] == 3:
                    image = image.rotate(180, expand=True)
                if exif[274] == 6:
                    image = image.rotate(270, expand=True)
                if exif[274] == 8:
                    image = image.rotate(90, expand=True)

        # If has an alpha channel, remove it for consistency
        if img_shape[-1] == 4:
            image = image[..., :3]

        return np.array(image)

    def auto_download(self, dataDir, dataType, dataYear):
        """TODO: Download the TACO dataset/annotations if requested."""

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        image_info = self.image_info[image_id]

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id("taco.{}".format(annotation['category_id']))
            if class_id:
                m = utilsTACO.annToMask(annotation, image_info["height"],image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool_)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super().load_mask(image_id)

    def replace_dataset_classes(self, dataset, class_map):
        """ Replaces classes of dataset based on a dictionary"""
        class_new_names = list(set(class_map.values()))
        class_new_names.sort()
        class_originals = copy.deepcopy(dataset['categories'])
        dataset['categories'] = []
        class_ids_map = {}  # map from old id to new id

        # Assign background id 0
        has_background = False
        if 'Background' in class_new_names:
            if class_new_names.index('Background') != 0:
                class_new_names.remove('Background')
                class_new_names.insert(0, 'Background')
            has_background = True

        # Replace categories
        for id_new, class_new_name in enumerate(class_new_names):

            # Make sure id:0 is reserved for background
            id_rectified = id_new
            if not has_background:
                id_rectified += 1

            category = {
                'supercategory': '',
                'id': id_rectified,  # Background has id=0
                'name': class_new_name,
            }
            dataset['categories'].append(category)
            # Map class names
            for class_original in class_originals:
                if class_map[class_original['name']] == class_new_name:
                    class_ids_map[class_original['id']] = id_rectified

        # Update annotations category id tag
        for ann in dataset['annotations']:
            ann['category_id'] = class_ids_map[ann['category_id']]
    def load_ann(self, image_id):
        return self.image_info[image_id]["annotations"] 