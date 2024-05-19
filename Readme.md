# How to use

The find.py script is a simple script that allows you to localize you in a map given 4 points in an image of reference and the same 4 points in the map.

To use the script run the following command:

```bash
python find.py -i <image> -c <calibration_file>
```

Where `<image>` is the path to the image of reference and `<calibration_file>` is the path to the calibration file with the K matrix and the distortion coefficients.

There is an example of the calibration file in the file `calibration` and the correspoding image of reference is `image.jpeg`.