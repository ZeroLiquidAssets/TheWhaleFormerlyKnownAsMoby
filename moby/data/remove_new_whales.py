import os
import pandas as pd

from logzero import logger

from moby.constants import DATA_DIR


def main(output_dir):
    train = pd.read_csv(DATA_DIR / "raw" / "train.csv")
    raw_train_images = pd.read_csv(DATA_DIR / "raw" / "train")

    # Remove "new_whale"
    new_whales = train.query('Id == "new_whale"')
    train_no_new_whales = train.drop(new_whales.index)
    
    n_diff = train.shape[0] - train_no_new_whales.shape[0]

    # Write the training file
    train_no_new_whales.to_csv(output_dir / "train.csv", index=False)

    # Move images
    output_img_dir = output_dir / "train"
    for img in train_no_new_whales["Image"].unique():
        src = raw_train_images / img
        target = output_img_dir / img
        logger.debug(
            "Copying %(src)s to %(target)s", 
            {"src": src, "target", target}
        )
        os.copy(src, target)
    logger.info("Removed %(n)d 'new_whale' from train.csv", {"n": n_diff})
    logger.info("Copied %(n)d non-'new_whale' images from %(src)s to %(target)s",
        {"n": train_no_new_whales.shape[0], "src": raw_train_images, "target": output_img_dir}
    )


if __name__ == "__main__":
    output_dir = DATA_DIR / "interim" / "1_no_new_whales"    
    main(output_dir)
