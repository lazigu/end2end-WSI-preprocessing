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
import random
import csv
import cv2
import time
from datetime import timedelta
from pathlib import Path
import torch
from helpers import stainNorm_Macenko
from helpers.common import supported_extensions
from helpers.concurrent_canny_rejection import reject_background
from helpers.loading_slides import process_slide_jpg, load_slide, get_raw_tile_list, read_annotations
from helpers.feature_extractors import FeatureExtractor
from marugoto.marugoto.extract.extract import extract_features_
from shapely.geometry import box
from shapely.affinity import scale
import matplotlib.pyplot as plt
import shapely.geometry as sg
import numpy as np
from matplotlib.patches import Patch
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Normalise WSI directly.')

    parser.add_argument('-o', '--output-path', type=Path, required=True,
                        help='Path to save features to.')
    parser.add_argument('--wsi-dir', metavar='DIR', type=Path, required=True,
                        help='Path of where the whole-slide images are.')
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
    parser.add_argument('--roi-dir', metavar='DIR', type=Path, required=False,
                        help='Path of where CSV files containing ROI coordinates are.', default=None)

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
    roi_path = args.roi_dir
    print(f"ROI dir path was given: {roi_path}")
    norm=args.norm

    if has_gpu:
        print(f"Number of GPUs in the system: {torch.cuda.device_count()}")

    if norm:
        print("\nInitialising Macenko normaliser...")
        target = cv2.imread('normalization_template.jpg') #TODO: make scaleable with path
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        normalizer = stainNorm_Macenko.Normalizer()
        normalizer.fit(target)
        logging.info('Running WSI to normalised feature extraction...')
    else:
        logging.info('Running WSI to raw feature extraction...')

    #initialize the feature extraction model
    print(f"\nInitialising {args.extractor} model...")
    extractor = FeatureExtractor(args.extractor)
    model, model_name = extractor.init_feat_extractor(checkpoint_path=args.model)

    #create output feature folder, f.e.:
    #~/output_folder/E2E_macenko_xiyuewang-ctranspath/
    (args.output_path).mkdir(parents=True, exist_ok=True)
    
    norm_method = "E2E_macenko_" if args.norm else "E2E_raw_"
    model_name_norm = Path(norm_method+model_name)
    output_file_dir = args.output_path/model_name_norm
    output_file_dir.mkdir(parents=True, exist_ok=True)
    
    total_start_time = time.time()
    
    img_name = "norm_slide.jpg" if args.norm else "canny_slide.jpg"

    img_dir = sum((list(args.wsi_dir.glob(f'**/*.{ext}'))
                for ext in supported_extensions),
                start=[])
                       
    for slide_url in (progress := tqdm(img_dir, leave=False)):
        
        if not args.only_fex:
            slide_name = Path(slide_url).stem
            slide_cache_dir = args.cache_dir/slide_name
            slide_cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            slide_name = Path(slide_url).parent.name

        progress.set_description(slide_name)
        
        feat_out_dir = output_file_dir/slide_name

        if not (os.path.exists((f'{feat_out_dir}.h5'))):
            # Load WSI as one image
            logging.info(f"\nLoading {slide_name}")
            try:
                slide = openslide.OpenSlide(str(slide_url))
            except openslide.lowlevel.OpenSlideUnsupportedFormatError:
                logging.error(f"Unsupported format for {slide_name}")
                continue
            except Exception as e:
                logging.error(f"Failed loading {slide_name}, error: {e}")
                continue

            #measure time performance
            start_time = time.time()
            slide_array, slide_mpp = load_slide(slide=slide, cores=args.cores)
            if slide_array is None:
                if args.del_slide:
                    print(f"Skipping slide and deleting {slide_url} due to missing MPP...")
                    os.remove(str(slide_url))
                continue
            #save raw .svs jpg
            (PIL.Image.fromarray(slide_array)).save(f'{slide_cache_dir}/slide.jpg')

            #remove .SVS from memory
            del slide

            print("\n--- Loaded slide: %s seconds ---" % (time.time() - start_time))
            #########################

            #########################
            #Do edge detection here and reject unnecessary tiles BEFORE normalisation
            bg_reject_array, rejected_tile_array, patch_shapes = reject_background(img = slide_array, patch_size=(224,224), step=224, outdir=args.cache_dir, save_tiles=False, cores=args.cores)

            #measure time performance
            start_time = time.time()
            #pass raw slide_array for getting the initial concentrations, bg_reject_array for actual normalisation
            if norm:
                slide_jpg = slide_cache_dir/"norm_slide.jpg"
                logging.info(f"Normalising {slide_name}...")
                canny_img, img_norm_wsi_jpg, canny_norm_patch_list, coords_list = normalizer.transform(slide_array, bg_reject_array, rejected_tile_array, patch_shapes, cores=args.cores)
                print(f"\n--- Normalised slide {slide_name}: {(time.time() - start_time)} seconds ---")
                img_norm_wsi_jpg.save(slide_jpg) #save WSI.svs -> WSI.jpg

            else:
                canny_img, canny_norm_patch_list, coords_list = get_raw_tile_list(slide_array.shape, bg_reject_array, rejected_tile_array, patch_shapes)

            print("Saving Canny background rejected image...")
            canny_img.save(f'{slide_cache_dir}/canny_slide.jpg')

            #remove original slide jpg from memory
            del slide_array

            #optionally removing the original slide from harddrive
            if args.del_slide:
                print(f"Deleting slide {slide_name} from local folder...")
                os.remove(str(slide_url))

            if roi_path is not None and os.path.exists(os.path.join(roi_path, str(slide_name) + ".csv")):

                if not os.path.exists(os.path.join(args.cache_dir, "plots")):
                    os.makedirs(os.path.join(args.cache_dir, "plots"))
                if not os.path.exists(os.path.join(args.cache_dir, "tiles_coords")):
                    os.makedirs(os.path.join(args.cache_dir, "tiles_coords"))

                tiles_within_roi = []
                coords_of_remaining_tiles = []
                tile_boxes = []
                print("CSV file with coordinates of ROIs is found. Loading annotations...")
                try:
                    annPolys, rectcoords = read_annotations(os.path.join(roi_path, str(slide_name) + ".csv"))
                except Exception as err:
                    print(f"CSV file could not be successfully read (error: {err}), slide {slide_name} will be skipped...")
                    continue

                intersection_threshold = 0.6

                target_mpp = 256/224
                if slide_mpp is None:
                    print('Slide MPP could not be determined, please run the pipeline from scratch...')
                    continue

                scale_factor = slide_mpp / target_mpp
                scaled_annPolys = scale(annPolys, xfact=scale_factor, yfact=scale_factor, origin=(0, 0))

                with open(os.path.join(args.cache_dir, "tiles_coords", slide_name + '.csv'), 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["Coords"])

                    for i, tile in tqdm(enumerate(canny_norm_patch_list), desc="Checking whether tiles are within ROIs", total=len(canny_norm_patch_list)):

                        tile_size_x = tile.shape[1]
                        tile_size_y = tile.shape[0]

                        # create a bounding box for each tile - box(minx, miny, maxx, maxy)
                        tile_box = box(coords_list[i][0], coords_list[i][1],
                                       coords_list[i][0] + tile_size_x, coords_list[i][1] + tile_size_y)
                        tile_boxes.append(tile_box)
                        writer.writerow([tile_box])

                        try:
                            if isinstance(scaled_annPolys, sg.MultiPolygon):
                                for single_polygon in scaled_annPolys.geoms:
                                    if single_polygon.intersects(tile_box):
                                        intersection = single_polygon.intersection(tile_box)
                                        if (intersection.area / tile_box.area) >= 0.6:
                                            tiles_within_roi.append(tile)
                                            coords_of_remaining_tiles.append(coords_list[i])
                            else:
                                if scaled_annPolys.intersects(tile_box):
                                    intersection = scaled_annPolys.intersection(tile_box)
                                    if (intersection.area / tile_box.area) >= 0.6:
                                        tiles_within_roi.append(tile)
                                        coords_of_remaining_tiles.append(coords_list[i])
                        except Exception as err:
                            print(f"There was an error with a polygon, which was not caught by existing checks, skipping this tile. Error: {err}")
                            continue


                print(f"{len(tiles_within_roi)} tiles remain out of {len(canny_norm_patch_list)}")

                fig, ax = plt.subplots()
                if isinstance(scaled_annPolys, sg.MultiPolygon):
                    for i, polygon in enumerate(scaled_annPolys.geoms):
                        x, y = polygon.exterior.xy
                        ax.fill(x, y, alpha=0.5, fc='r', label=f'Annotated Tissue')
                else:
                    x, y = scaled_annPolys.exterior.xy
                    ax.fill(x, y, alpha=0.5, fc='r', ec='none')
                for tile in tile_boxes:
                    x, y = tile.exterior.xy
                    ax.fill(x, y, alpha=0.5, fc='b', ec='none', label='Extracted Tiles')

                legend_elements = [Patch(facecolor='red', edgecolor='r', alpha=0.5, label='Annotated Tissue'),
                                   Patch(facecolor='blue', edgecolor='b', alpha=0.5, label='Extracted Tiles')]
                ax.legend(handles=legend_elements, loc='upper right')
                ax.invert_yaxis()
                plt.savefig(os.path.join(args.cache_dir, "plots", slide_name + ".png"))

            else:
                print(f"No CSV file with ROIs was found, skipping this slide {slide_name}...")
                continue

            print(f"Extracting {args.extractor} features from {slide_name}")
            #FEATURE EXTRACTION
            #measure time performance
            start_time = time.time()
            if len(tiles_within_roi) > 0:
                extract_features_(model=model, model_name=model_name, norm_wsi_img=tiles_within_roi,
                                coords=coords_of_remaining_tiles, wsi_name=slide_name, outdir=feat_out_dir, cores=args.cores, is_norm=args.norm)
                print("\n--- Extracted features from slide: %s seconds ---" % (time.time() - start_time))
            else:
                print(f"0 tiles remain to extract features from after pre-processing {slide_name}, skipping...")
                continue
            #########################

        else:
            print(f"{slide_name}.h5 already exists. Skipping...")
            if args.del_slide:
                print(f"Deleting slide {slide_name} from local folder...")
                os.remove(str(slide_url))

    print(f"--- End-to-end processing time of {len(img_dir)} slides: {str(timedelta(seconds=(time.time() - total_start_time)))} ---")
