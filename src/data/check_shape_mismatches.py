import argparse
from pathlib import Path
from PIL import Image

def check_image_mask_shapes(
    dataset_path,
    img_folder='frames',
    mask_folder='masks',
    mask_suffix=''
):
    dataset_path = Path(dataset_path).resolve()
    img_path = dataset_path / img_folder
    mask_path = dataset_path / mask_folder

    img_files = sorted(list(img_path.glob('*.png')) + list(img_path.glob('*.jpg')))

    if not img_files:
        print(f"No image files found in {img_path}")
        return

    mismatch_count = 0
    pairs_fine = 0
    missing_mask_count = 0

    for img_file in img_files:
        # Construct mask filename with optional suffix
        mask_file = mask_path / (img_file.stem + mask_suffix + img_file.suffix)

        # Check if mask exists
        if not mask_file.exists():
            print(f"[NO MASK FOUND] {img_file.name} --> expected mask: {mask_file.name}")
            missing_mask_count += 1
            continue

        # Open image and mask
        with Image.open(img_file) as img, Image.open(mask_file) as msk:
            img_size = img.size  # (width, height)
            msk_size = msk.size  # (width, height)

            if img_size != msk_size:
                print(f"[SIZE MISMATCH] {img_file.name}: image={img_size} mask={mask_file.name}={msk_size}")
                mismatch_count += 1
            else:
                pairs_fine += 1

    total_checked = len(img_files)
    print("\nSUMMARY:")
    print(f"  Total images checked: {total_checked}")
    print(f"  Missing masks: {missing_mask_count}")
    print(f"  Mismatched image/mask pairs: {mismatch_count}")
    print(f"  Perfectly matched pairs: {pairs_fine}")

def main():
    parser = argparse.ArgumentParser(description="Check that image and mask shapes match.")
    parser.add_argument('--path', '-p', required=True,
                        help="Path to the dataset directory.")
    parser.add_argument('--img_folder', default='frames',
                        help="Subfolder for images.")
    parser.add_argument('--mask_folder', default='masks',
                        help="Subfolder for masks.")
    parser.add_argument('--mask_suffix', default='',
                        help="Optional suffix for mask files (e.g. '_bolus').")
    args = parser.parse_args()

    check_image_mask_shapes(
        dataset_path=args.path,
        img_folder=args.img_folder,
        mask_folder=args.mask_folder,
        mask_suffix=args.mask_suffix
    )

if __name__ == '__main__':
    main()
