import os
import logging
import pandas as pd

from logzero import logger

from moby.constants import DATA_DIR


logger.setLevel(logging.INFO)


def main(output_dir):
    """Re-create train.csv without new_whale, and creates a copy of all non-new_whale images.
    
    Args:
        output_dir (pathlib.Path): pathlib.Path object to the target directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA_DIR / "raw" / "train.csv")
    raw_train_images = DATA_DIR / "raw" / "train"

    # Remove "new_whale"
    new_whales = train.query('Id == "new_whale"')
    train_no_new_whales = train.drop(new_whales.index)

    n_diff = train.shape[0] - train_no_new_whales.shape[0]

    # Write the training file
    train_no_new_whales.to_csv(output_dir / "train.csv", index=False)

    # Move images
    output_img_dir = output_dir / "train"
    output_img_dir.mkdir(parents=True, exist_ok=True)

    for img in train_no_new_whales["Image"].unique():
        src = raw_train_images / img
        target = output_img_dir / img
        # logger.debug("Linking %(src)s to %(target)s", {"src": src, "target": target})
        os.symlink(src, target)
    logger.info("Removed %(n)d 'new_whale' from train.csv", {"n": n_diff})
    logger.info(
        "Linked %(n)d non-'new_whale' images from %(src)s to %(target)s",
        {
            "n": train_no_new_whales.shape[0],
            "src": raw_train_images,
            "target": output_img_dir,
        },
    )


default_output_dir = DATA_DIR / "interim" / "1_no_new_whales"

if __name__ == "__main__":
    """Script mode for development.
    """
    main(default_output_dir)
