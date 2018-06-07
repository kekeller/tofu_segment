#!/bin/bash

wget -nd -r -P ./ -A tif,png http://kekeller.com/data.tar
tar xvC ./ -f data.tar
