#!/bin/bash

wget -nd -r -P ./data/train_images -A tif http://kekeller.com/IMAGES/
wget -nd -r -P ./data/train_masks -A tif http://kekeller.com/MASKS/