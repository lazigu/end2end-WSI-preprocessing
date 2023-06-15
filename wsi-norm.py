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
import numpy as np
import openslide
from tqdm import tqdm
import fractions
import PIL
import cv2
import time
from datetime import timedelta
from pathlib import Path
import torch
from scipy import ndimage
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
    parser.add_argument('-z','--zoom',action='store_true', 
                        default=False,help="Extract features for different zoom levels.")
    parser.set_defaults(norm=True)
    parser.add_argument('-d', '--del-slide', action='store_true', default=False,
                         help='Removing the original slide after processing.')
    parser.add_argument('--only-fex', action='store_true', default=False)
    parser.add_argument('--target-mpp', type=str, default="256/224")
    parser.add_argument('--base-pxl-size', type=int, default=224)
    
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

    if "/" in args.target_mpp:
        args.target_mpp = float(args.target_mpp.split('/')[0]) / float(args.target_mpp.split('/')[1]) 
    else:
        args.target_mpp = float(args.target_mpp)
    
    print(f"Target MPP: {args.target_mpp:.6f}")
    
    #create output feature folder, f.e.:
    #~/output_folder/E2E_macenko_xiyuewang-ctranspath/
    (args.output_path).mkdir(parents=True, exist_ok=True)
    
    norm_method = "E2E_macenko_" if args.norm else "E2E_raw_"
    model_name_norm = Path(norm_method+model_name)
    output_file_dir = args.output_path/model_name_norm
    output_file_dir.mkdir(parents=True, exist_ok=True)
    
    total_start_time = time.time()
    
    img_name = "norm_slide.jpg" if args.norm else "canny_slide.jpg"
    if not args.only_fex:
        img_dir = sum((list(args.wsi_dir.glob(f'**/*.{ext}'))
                    for ext in supported_extensions),
                    start=[])
        img_dir = np.random.permutation(img_dir).tolist()
    else:
        img_dir = list(args.wsi_dir.glob(f'**/*/{img_name}'))
        img_dir = np.random.permutation(img_dir).tolist()
                       
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
            if (args.only_fex and (slide_jpg := slide_url).exists()) \
                or (slide_jpg := slide_cache_dir/'norm_slide.jpg').exists():
                canny_norm_patch_list, coords_list, zoom_list, patch_saved, total = process_slide_jpg(slide_jpg, zoom=args.zoom, cores=args.cores, base_pxl_size=args.base_pxl_size)
                print(f"Loaded {img_name}, {patch_saved}/{total} tiles remain")
            else:
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
                slide_array = load_slide(slide,target_mpp=args.target_mpp, cores=args.cores)
                if slide_array is None:
                    if args.del_slide:
                        print(f"Skipping slide and deleting {slide_url} due to missing MPP...")
                        os.remove(str(slide_url))
                    continue
                #save raw .svs jpg
                if slide_array.shape[0] > 65500 or slide_array.shape[1] > 65500:
                     slide_array = ndimage.zoom(slide_array, (min(65500,slide_array.shape[0]) / slide_array.shape[0],
                                                              min(65500,slide_array.shape[1]) / slide_array.shape[1], 1))
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
                    logging.info(f"Normalising {slide_name}...")
                    canny_img, img_norm_wsi_jpg, canny_norm_patch_list, coords_list, zoom_list = normalizer.transform(slide_array, bg_reject_array, rejected_tile_array, patch_shapes, cores=args.cores)
                    print(f"\n--- Normalised slide {slide_name}: {(time.time() - start_time)} seconds ---")
                    img_norm_wsi_jpg.save(slide_jpg) #save WSI.svs -> WSI.jpg

                else:
                    canny_img, canny_norm_patch_list, coords_list, zoom_list = get_raw_tile_list(slide_array.shape, bg_reject_array, rejected_tile_array, patch_shapes)

                print("Saving Canny background rejected image...")
                canny_img.save(f'{slide_cache_dir}/canny_slide.jpg')
                
                #remove original slide jpg from memory
                del slide_array
                
                #optionally removing the original slide from harddrive
                if args.del_slide:
                    print(f"Deleting slide {slide_name} from local folder...")
                    os.remove(str(slide_url))

                #img_norm_wsi_jpg.save(slide_jpg) #save WSI.svs -> WSI.jpg

            print(f"Extracting {args.extractor} features from {slide_name}")
            #FEATURE EXTRACTION
            #measure time performance
            start_time = time.time()
            # if "zoom_list" not in locals():
            #     zoom_list = np.ones(len(coords_list)).tolist()
            extract_features_(model=model, model_name=model_name, norm_wsi_img=canny_norm_patch_list,
                               coords=coords_list, zoom=zoom_list, wsi_name=slide_name, outdir=feat_out_dir, cores=args.cores, is_norm=args.norm)
            print("\n--- Extracted features from slide: %s seconds ---" % (time.time() - start_time))
            #########################

        else:
            print(f"{slide_name}.h5 already exists. Skipping...")
            if args.del_slide:
                print(f"Deleting slide {slide_name} from local folder...")
                os.remove(str(slide_url))

    print(f"--- End-to-end processing time of {len(img_dir)} slides: {str(timedelta(seconds=(time.time() - total_start_time)))} ---")
