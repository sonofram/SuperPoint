import numpy as np
import tensorflow as tf
from pathlib import Path
import os

from .base_dataset import BaseDataset
from .utils import pipeline
from superpoint.settings import DATA_PATH, EXPER_PATH, TMPDIR


class Synth(BaseDataset):
    default_config = {
        'labels': None,
        'cache_in_memory': False,
        'validation_size': 100,
        'truncate': None,
        'preprocessing': {
            'resize': None, #[240, 320]
        },
        'num_parallel_calls': 10,
        'augmentation': {
            'photometric': {
                'enable': False,
                'primitives': 'all',
                'params': {},
                'random_order': True,
            },
            'homographic': {
                'enable': False,
                'params': {},
                'valid_border_margin': 0,
            },
        },
        'warped_pair': {
            'enable': False,
            'params': {},
            'valid_border_margin': 0,
        },
    }

    def _init_dataset(self, **config):
        
        shape_dir_list = ['draw_checkerboard','draw_cube','draw_ellipses',
                  'draw_lines','draw_multiple_polygons','draw_star',
                  'draw_stripes']
        img_dir = '/images/test/'
        p_dir   = '/points/test/'
        ifiles = []
        pfiles = []
        outputdirs = []
        for sdir in shape_dir_list:
            idir = TMPDIR+'/'+sdir+img_dir
            pdir = TMPDIR+'/'+sdir+p_dir
            ifiles_list = [f for f in os.listdir(idir)]
            pfiles_list = [f.replace(".png",".npy") for f in ifiles_list]
            ifiles_list = [os.path.join(idir, f) for f in ifiles_list]
            pfiles_list = [os.path.join(pdir, f) for f in pfiles_list]
                            
            # Accumulate images in list for later operations.
            if ifiles == None:
                ifiles = ifiles_list
                pfiles = pfiles_list
                outputdirs = [sdir+'/'+p_dir] * len(ifiles_list)
            else:    
                ifiles = ifiles + ifiles_list
                pfiles = pfiles + pfiles_list
                outputdirs = outputdirs + [sdir+'/'+p_dir] * len(ifiles_list)
            
            
        names = [Path(p).stem for p in ifiles]
        files = {'image_paths': ifiles, 'names': names, 'label_paths': pfiles, 'output_dirs': outputdirs}


        tf.data.Dataset.map_parallel = lambda self, fn: self.map(
                fn, num_parallel_calls=config['num_parallel_calls'])

        return files

    def _get_data(self, files, split_name, **config):
        has_keypoints = 'label_paths' in files

        def _read_image(path):
            image = tf.read_file(path)
            image = tf.image.decode_png(image, channels=3)
            return tf.cast(image, tf.float32)

        def _preprocess(image):
            image = tf.image.rgb_to_grayscale(image)
            if config['preprocessing']['resize']:
                image = pipeline.ratio_preserving_resize(image,
                                                         **config['preprocessing'])
            return image

        # Python function
        def _read_points(filename):
            return np.load(filename.decode('utf-8')).astype(np.float32)

        outputs = tf.data.Dataset.from_tensor_slices(files['output_dirs'])
        names = tf.data.Dataset.from_tensor_slices(files['names'])
        images = tf.data.Dataset.from_tensor_slices(files['image_paths'])
        images = images.map(_read_image)
        images = images.map(_preprocess)
        data = tf.data.Dataset.zip({'image': images, 'name': names, 'output_dirs': outputs})

        # Add keypoints
        if has_keypoints:
            kp = tf.data.Dataset.from_tensor_slices(files['label_paths'])
            kp = kp.map(lambda path: tf.py_func(_read_points, [path], tf.float32))
            kp = kp.map(lambda points: tf.reshape(points, [-1, 2]))
            data = tf.data.Dataset.zip((data, kp)).map(
                    lambda d, k: {**d, 'keypoints': k})
            data = data.map(pipeline.add_dummy_valid_mask)

        # Generate the keypoint map
        if has_keypoints:
            data = data.map_parallel(pipeline.add_keypoint_map)
        data = data.map_parallel(
            lambda d: {**d, 'image': tf.to_float(d['image']) / 255.})
        
        return data
