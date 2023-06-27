__author__ = "Omar El Nahhas"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Omar"]
__email__ = "omar.el_nahhas@tu-dresden.de"


import argparse
from pathlib import Path
import logging
import os
import openslide
from tqdm import tqdm
import PIL
import cv2
import time
from datetime import timedelta
from pathlib import Path
import torch
from helpers import stainNorm_Macenko
from helpers.common import supported_extensions
from helpers.concurrent_canny_rejection import reject_background
from helpers.loading_slides import process_slide_jpg, load_slide, get_raw_tile_list
from helpers.feature_extractors import FeatureExtractor
from marugoto.marugoto.extract.extract import extract_features_

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Normalise WSI directly.')

    parser.add_argument('-o', '--output-path', type=Path, required=True,
                        help='Path to save features to.')
    parser.add_argument('--tiles-dir', metavar='DIR', type=Path, required=True,
                        help='Path of where the tiles are.')
    parser.add_argument('-m', '--model', metavar='DIR', type=Path, required=True,
                        help='Path of where model for the feature extractor is.')
    parser.add_argument('--cache-dir', type=Path, default=None,
        help='Directory to store resulting slide JPGs.')
    parser.add_argument('-e', '--extractor', type=str, 
                        help='Feature extractor to use.')
    parser.add_argument('-c', '--cores', type=int, default=8,
                    help='CPU cores to use, 8 default.')
    parser.add_argument('-n','--norm', action='store_true')
    parser.add_argument('--no-norm', dest='norm', action='store_false')
    parser.set_defaults(norm=True)
    parser.add_argument('-d', '--del-slide', action='store_true', default=False,
                         help='Removing the original slide after processing.')
    parser.add_argument('--only-fex', action='store_true', default=False)

    args = parser.parse_args()


PIL.Image.MAX_IMAGE_PIXELS = None

if __name__ == "__main__":
    # print current dir
    print(f"Current working directory: {os.getcwd()}")
    Path(args.cache_dir).mkdir(exist_ok=True, parents=True)
    logdir = args.cache_dir/'logfile'
    logging.basicConfig(filename=logdir, force=True)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(f'Stored logfile in {logdir}')
    #init the Macenko normaliser
    print(f"Number of CPUs in the system: {os.cpu_count()}")
    print(f"Number of CPU cores used: {args.cores}")
    has_gpu=torch.cuda.is_available()
    print(f"GPU is available: {has_gpu}")
    norm=args.norm

    if has_gpu:
        print(f"Number of GPUs in the system: {torch.cuda.device_count()}")

    # initialize the feature extraction model
    print(f"\nInitialising {args.extractor} model...")
    extractor = FeatureExtractor(args.extractor)
    model, model_name = extractor.init_feat_extractor(checkpoint_path=args.model)

    # create output feature folder, f.e.:
    # ~/output_folder/E2E_macenko_xiyuewang-ctranspath/
    (args.output_path).mkdir(parents=True, exist_ok=True)

    norm_method = "E2E_macenko_" if args.norm else "E2E_raw_"
    model_name_norm = Path(norm_method + model_name)
    output_file_dir = args.output_path / model_name_norm
    print(output_file_dir)
    output_file_dir.mkdir(parents=True, exist_ok=True)

    total_start_time = time.time()

    slide_dir = os.listdir(args.tiles_dir)

    for slide_url in (progress := tqdm(slide_dir, leave=False)):

        slide_name = os.path.splitext(slide_url)[0]
        print(f"slide name: {slide_name}")

        progress.set_description(slide_name)
        
        feat_out_dir = output_file_dir/slide_name

        if not (os.path.exists((f'{feat_out_dir}.h5'))):

            logging.info(f"\nLoading tiles of {slide_name}")
 
            # measure time performance
            start_time = time.time()

            tiles_list = []
            coords_list = []

            tiles_fnames = os.listdir(os.path.join(args.tiles_dir, slide_url))
            for fname in tqdm(tiles_fnames, desc="Loading tiles"):
                if fname.endswith('.jpg'):
                    # Load the tile and append it to the tiles list
                    tile_path = os.path.join(os.path.join(args.tiles_dir, slide_url, fname))
                    # print(tile_path)
                    tile = cv2.imread(tile_path)
                    tiles_list.append(tile)

                    # Extract the coordinates from the filename
                    coords_str = fname.split('_')[-1].split('.jpg')[0]
                    coords_str = coords_str.replace('(', '').replace(')', '')
                    x, y = map(int, coords_str.split(','))
                    coordinate_tuple = (x, y)
                    coords_list.append(coordinate_tuple)


            if len(tiles_list) == 0:
                print(f"Skipping slide {slide_url} because no tiles could be found/loaded correctly...")
                continue

            print(f"\n--- Loaded {len(tiles_list)} tiles of the slide: %s seconds ---" % (time.time() - start_time))
            #########################

            print(f"Extracting {args.extractor} features from {slide_name}")
            #FEATURE EXTRACTION
            #measure time performance
            start_time = time.time()
            if len(tiles_list) > 0:
                extract_features_(model=model, model_name=model_name, norm_wsi_img=tiles_list,
                                coords=coords_list, wsi_name=slide_name, outdir=feat_out_dir, cores=args.cores, is_norm=args.norm)
                print(f"\n--- Extracted features from slide {slide_name}: %s seconds ---" % (time.time() - start_time))
            else:
                print("0 tiles remain to extract features from after pre-processing {slide_name}, skipping...")
                continue
            #########################

        else:
            print(f"{slide_name}.h5 already exists. Skipping...")
            if args.del_slide:
                print(f"Deleting slide {slide_name} from local folder...")
                os.remove(str(slide_url))

    print(f"--- End-to-end processing time of {len(slide_dir)} slides: {str(timedelta(seconds=(time.time() - total_start_time)))} ---")