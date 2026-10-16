# Camera Calibration

Place at least 10 JPEG checkerboard images in `image/`, then run from this directory:

```bash
python -m pip install -r requirements.txt
python run_calibration.py --conditions Z0 Z_gamma CV5 CV0 LM0 LM2 CV2
```

Results and visualizations are saved to `outputs/`.
