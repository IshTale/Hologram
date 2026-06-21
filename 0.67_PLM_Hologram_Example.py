from TIPLMSuiteHologram import CGHGenerator, DeviceLibrary
import cv2
import numpy as np
from pathlib import Path

try:
        import torch
except ImportError:
        torch = None

# create instance of TI CGHGenerator and DeviceLibrary
G = CGHGenerator()
D = DeviceLibrary()

# Define input parameters
ViewImage1 = "./Bear1.png"   # replace with first view image
ViewImage2 = "./Bear2.png"                       # replace with second view image
RandomSeed = 123
NumIter = 500
ViewShiftPixels = 330  # Near image_width / 4 for maximum wrapped-order separation on the 1358-wide PLM.
PreparedDir = "./angle_mux_prepared"
FitMode = "contain"  # Preserve each photo's aspect ratio inside its order window. Options: "stretch", "crop", "contain".
DitherMethod = "floyd_steinberg"  # Options: "floyd_steinberg", "jarvis", "atkinson", "ordered".
# Keep the order window narrow enough that the two angular orders do NOT overlap:
#   OrderWindowWidthFraction < viewShiftPixels / (image_width / 2) = 330 / 679 ~= 0.49.
# 0.62 (the old value) violates this, so the two views bleed into each other.
# With ConfineTargetsToOrderWindow=True the whole image is fit inside this window,
# so a narrow window no longer "cuts content" -- do NOT raise it toward 1.0.
OrderWindowWidthFraction = 0.38
OrderWindowHeightFraction = 1.0
OrderWindowFeatherFraction = 0.04
OutsideOrderWeight = 0.25  # Pushes light away from the left/right stray orders.
ConfineTargetsToOrderWindow = True  # Fit the whole image into the kept order window (full image per view, no overlap).

if(RandomSeed is not None):
        np.random.seed(int(RandomSeed))
        if(torch is not None):
                torch.manual_seed(int(RandomSeed))

# Call create angle-multiplexed hologram.
# The two source images are stretched to 1358x800 and saved as photo-halftoned 1-bit BMPs.
G.createAngleMultiplexedCGH(DeviceDictionary=D.defineDevice('0.67'), # Device Dictionary
                            filenames=[ViewImage1, ViewImage2], # Two input view images
                            outputPreparedDir=PreparedDir, # Stores 1358x800 1-bit BMP inputs
                            viewShiftPixels=ViewShiftPixels, # Symmetric angular carrier separation
                            numIter=NumIter, learningRate=0.08,
                            threshold=0.5, dither=True,
                            ditherMethod=DitherMethod, fitMode=FitMode,
                            blackPercentile=1.0, whitePercentile=99.0,
                            gamma=0.9, sharpenAmount=0.35,
                            FlipUD=True, # 0.67 EVM requires an image flip.
                            ShiftFOV=True,
                            orderWindowWidthFraction=OrderWindowWidthFraction,
                            orderWindowHeightFraction=OrderWindowHeightFraction,
                            orderWindowFeatherFraction=OrderWindowFeatherFraction,
                            outsideOrderWeight=OutsideOrderWeight,
                            maskRecoveredOrders=True,
                            confineTargetsToOrderWindow=ConfineTargetsToOrderWindow,
                            showImages=False)

outputRoot = Path(ViewImage1).stem + "_TO_" + Path(ViewImage2).stem
G.writeCGHToFile(outputRoot + "_AngleMux_CGH.bmp") # Write the TI PLM mapped CGH to a file.

# Optional: Display both reconstructed angular views.
def display_normalize(image, high_percentile=99.5):
        image = np.asarray(image, dtype=np.float32)
        lo = float(np.percentile(image, 1.0))
        hi = float(np.percentile(image, high_percentile))
        if hi <= lo:
                return cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX)
        image = np.clip(image, lo, hi)
        return ((image - lo) / (hi - lo)).astype(np.float32)

cv2.destroyAllWindows()
cv2.imshow("Angle View 1", display_normalize(G.angleMuxRecovered_disc[0]))
cv2.imshow("Angle View 2", display_normalize(G.angleMuxRecovered_disc[1]))
cv2.waitKey(0)
