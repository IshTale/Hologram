import os
import numpy as np
import sys
from scipy.fftpack import dct, idct
import cv2 # For resizing images/image processing and IO
from tkinter import Tk
from tkinter.filedialog import askopenfilename # Prompt user to enter input file
from tqdm import tqdm # For progress bar
from PIL import Image # For image input/output

# Optional imports for additional GPU support and speedup
_TORCH_INSTALL_HINT = (
        "PyTorch is required for the ADAM, ADAMWGS, and MULTICOLOR algorithms. "
        "Install it from https://pytorch.org/get-started/locally/. "
        "For Windows + CUDA 12.8, use: "
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128"
)

try:
        import torch
        from torch import nn
except ImportError:
        torch = None

        class _MissingTorchNN():
                class Module():
                        pass

                @staticmethod
                def Parameter(*args, **kwargs):
                        raise ImportError(_TORCH_INSTALL_HINT)

        nn = _MissingTorchNN()

# Optional TI provided phase mappings
try:
       from ti_phase_light_modulator import PLM
except:
        pass
       #print("TI PLM module not found. Using default PLM mapping functions.")

class DeviceLibrary():

        # Device parameters
        discScheme = None
        pitchW = None
        pitchH = None
        w = None
        h = None
        nLevel = None
        pLevel = None
        memory_layout = None
        memory_lut = None
        evm_padding = None
        evm_fliplr = None
        evm_flipud = None
        _VISIBLE_LUT_REFERENCE_WAVELENGTH_M = 532e-9
        _VISIBLE_EMPIRICAL_REFERENCE_PLEVEL = np.array(
                [0.0000, 0.0118, 0.0242, 0.0464, 0.0665, 0.0823, 0.1295, 0.2018,
                 0.3069, 0.3384, 0.3941, 0.4730, 0.5546, 0.6309, 0.7738, 0.9375,
                 1.0000],
                dtype=np.float64
        )
        _VISIBLE_EMPIRICAL_NAMED_PLEVELS = {
                405: np.remainder(_VISIBLE_EMPIRICAL_REFERENCE_PLEVEL * (_VISIBLE_LUT_REFERENCE_WAVELENGTH_M / 405e-9), 1.0),
                480: np.remainder(_VISIBLE_EMPIRICAL_REFERENCE_PLEVEL * (_VISIBLE_LUT_REFERENCE_WAVELENGTH_M / 480e-9), 1.0),
                532: _VISIBLE_EMPIRICAL_REFERENCE_PLEVEL.copy(),
                540: np.remainder(_VISIBLE_EMPIRICAL_REFERENCE_PLEVEL * (_VISIBLE_LUT_REFERENCE_WAVELENGTH_M / 540e-9), 1.0),
                541: np.remainder(_VISIBLE_EMPIRICAL_REFERENCE_PLEVEL * (_VISIBLE_LUT_REFERENCE_WAVELENGTH_M / 541e-9), 1.0),
                650: np.remainder(_VISIBLE_EMPIRICAL_REFERENCE_PLEVEL * (_VISIBLE_LUT_REFERENCE_WAVELENGTH_M / 650e-9), 1.0),
                700: np.remainder(_VISIBLE_EMPIRICAL_REFERENCE_PLEVEL * (_VISIBLE_LUT_REFERENCE_WAVELENGTH_M / 700e-9), 1.0),
        }

        @staticmethod
        def _normalizeLUTWavelengthKey(lutWavelength):
                if(lutWavelength is None):
                        return None

                lutWavelength = float(lutWavelength)
                if(lutWavelength <= 0):
                        raise ValueError("lutWavelength must be positive when provided.")

                if(lutWavelength < 1e-6):
                        lutWavelength = lutWavelength * 1e9

                return int(round(lutWavelength))

        @classmethod
        def _getVisibleEmpiricalPhaseLevels(cls, lambdaGlobal = None, lambdaM = 532e-9, lutWavelength = None):
                if(lutWavelength is not None and lambdaGlobal is not None and lambdaGlobal > 0):
                        raise ValueError("Use either lutWavelength or lambdaGlobal for visible empirical LUT selection, not both.")

                if(lutWavelength is not None):
                        lutKey = cls._normalizeLUTWavelengthKey(lutWavelength)
                        namedLut = cls._VISIBLE_EMPIRICAL_NAMED_PLEVELS.get(lutKey)
                        if(namedLut is None):
                                raise ValueError(f"Unsupported visible LUT wavelength: {lutWavelength}. Supported values are 405, 480, 532, 540, 541, 650, and 700 nm.")
                        return namedLut.copy()

                if(lambdaGlobal is None or lambdaGlobal <= 0):
                        return cls._VISIBLE_EMPIRICAL_REFERENCE_PLEVEL.copy()

                pLevel = cls._VISIBLE_EMPIRICAL_REFERENCE_PLEVEL * lambdaGlobal / lambdaM
                return np.remainder(pLevel, 1.0)

        # Define PLM device to be used. Returns a dictionary that describes the device
        def defineDevice(self, device, discScheme = "Empirical", lambdaGlobal = None, lambdaM = 532e-9, lutWavelength = None):
                # Define device
                device = device.upper()
                self.discScheme = discScheme

                # Define pixel pitch
                if(device == "0.47" or device == "0.67" or device == "0.98"):
                        self.pitchH = 10.8e-6
                        self.pitchW = 10.8e-6
                elif(device == "0.67_NIR_5BIT"):
                        self.pitchH = 10.8e-6
                        self.pitchW = 16.2e-6
                elif(device == "0.67_NIR_6BIT" or device == "0.67_NIR_5BIT"):
                        self.pitchH = 10.8e-6
                        self.pitchW = 16.2e-6
                elif(device == "0.47_SHRINK"):
                        self.pitchH = 5.4e-6
                        self.pitchW = 10.8e-6

                # Define array sizes
                if(device == "0.47"):
                        self.w = 960
                        self.h = 540
                        self.nLevel = 16
                elif(device == "0.67"):
                        self.w = 1280 # Can use 1280 or 1358 if using extended display
                        self.h = 800 # Can use 800 if using extended display
                        self.nLevel = 16
                elif(device == "0.98"):
                        self.w = 2048
                        self.h = 1088
                        self.nLevel = 16
                elif(device == "0.67_NIR_5BIT"):
                        self.w = 904
                        self.h = 800
                        self.nLevel = 32
                elif(device == "0.67_NIR_6BIT"):
                        self.w = 904
                        self.h = 800
                        self.nLevel = 64
                elif(device == "0.47_SHRINK"):
                        self.w = 2*960
                        self.h = 540
                        self.nLevel = 4

                # Define displacements
                if(discScheme == "Empirical" and (device == "0.47" or device == "0.67" or device == "0.98")):
                        # Visible empirical LUTs default to the nominal 532 nm table.
                        # An explicit lutWavelength selector can be used to switch
                        # between named tables such as 532 nm, 650 nm, and 700 nm.
                        self.pLevel = self._getVisibleEmpiricalPhaseLevels(lambdaGlobal, lambdaM, lutWavelength)
                elif(discScheme == "Empirical" and device == "0.67_NIR_5BIT"):
                        # Define odd and even column states
                        self.pLevel = np.zeros((2, 33))  # Create a 2x32 array for odd and even column states

                        # Odd column states
                        self.pLevel[0, 0] = 0.000  # Model 0.000
                        self.pLevel[0, 1] = 0.012  # Model 0.012
                        self.pLevel[0, 2] = 0.028  # Model 0.028
                        self.pLevel[0, 3] = 0.064  # Model 0.064
                        self.pLevel[0, 4] = 0.051  # Model 0.051
                        self.pLevel[0, 5] = 0.065  # Model 0.065
                        self.pLevel[0, 6] = 0.083  # Model 0.083
                        self.pLevel[0, 7] = 0.122  # Model 0.122
                        self.pLevel[0, 8] = 0.138  # Model 0.138
                        self.pLevel[0, 9] = 0.178  # Model 0.178
                        self.pLevel[0, 10] = 0.210  # Model 0.210
                        self.pLevel[0, 11] = 0.271  # Model 0.271
                        self.pLevel[0, 12] = 0.248  # Model 0.248
                        self.pLevel[0, 13] = 0.288  # Model 0.288
                        self.pLevel[0, 14] = 0.320  # Model 0.320
                        self.pLevel[0, 15] = 0.384  # Model 0.384
                        self.pLevel[0, 16] = 0.365  # Model 0.365
                        self.pLevel[0, 17] = 0.397  # Model 0.397
                        self.pLevel[0, 18] = 0.437  # Model 0.437
                        self.pLevel[0, 19] = 0.495  # Model 0.495
                        self.pLevel[0, 20] = 0.478  # Model 0.478
                        self.pLevel[0, 21] = 0.510  # Model 0.510
                        self.pLevel[0, 22] = 0.550  # Model 0.550
                        self.pLevel[0, 23] = 0.611  # Model 0.611
                        self.pLevel[0, 24] = 0.594  # Model 0.594
                        self.pLevel[0, 25] = 0.647  # Model 0.647
                        self.pLevel[0, 26] = 0.711  # Model 0.711
                        self.pLevel[0, 27] = 0.797  # Model 0.797
                        self.pLevel[0, 28] = 0.760  # Model 0.760
                        self.pLevel[0, 29] = 0.811  # Model 0.811
                        self.pLevel[0, 30] = 0.876  # Model 0.876
                        self.pLevel[0, 31] = 0.969  # Model 0.969
                        self.pLevel[0, 32] = 1.000

                        # Even column states
                        self.pLevel[1, 0] = 0.006  # Model 0.006
                        self.pLevel[1, 1] = 0.015  # Model 0.015
                        self.pLevel[1, 2] = 0.027  # Model 0.027
                        self.pLevel[1, 3] = 0.058  # Model 0.058
                        self.pLevel[1, 4] = 0.059  # Model 0.059
                        self.pLevel[1, 5] = 0.070  # Model 0.070
                        self.pLevel[1, 6] = 0.084  # Model 0.084
                        self.pLevel[1, 7] = 0.118  # Model 0.118
                        self.pLevel[1, 8] = 0.137  # Model 0.137
                        self.pLevel[1, 9] = 0.172  # Model 0.172
                        self.pLevel[1, 10] = 0.201  # Model 0.201
                        self.pLevel[1, 11] = 0.259  # Model 0.259
                        self.pLevel[1, 12] = 0.249  # Model 0.249
                        self.pLevel[1, 13] = 0.284  # Model 0.284
                        self.pLevel[1, 14] = 0.313  # Model 0.313
                        self.pLevel[1, 15] = 0.373  # Model 0.373
                        self.pLevel[1, 16] = 0.379  # Model 0.379
                        self.pLevel[1, 17] = 0.405  # Model 0.405
                        self.pLevel[1, 18] = 0.444  # Model 0.444
                        self.pLevel[1, 19] = 0.496  # Model 0.496
                        self.pLevel[1, 20] = 0.497  # Model 0.497
                        self.pLevel[1, 21] = 0.524  # Model 0.524
                        self.pLevel[1, 22] = 0.563  # Model 0.563
                        self.pLevel[1, 23] = 0.616  # Model 0.616
                        self.pLevel[1, 24] = 0.606  # Model 0.606
                        self.pLevel[1, 25] = 0.650  # Model 0.650
                        self.pLevel[1, 26] = 0.712  # Model 0.712
                        self.pLevel[1, 27] = 0.787  # Model 0.787
                        self.pLevel[1, 28] = 0.779  # Model 0.779
                        self.pLevel[1, 29] = 0.818  # Model 0.818
                        self.pLevel[1, 30] = 0.877  # Model 0.877
                        self.pLevel[1, 31] = 0.951  # Model 0.951
                        self.pLevel[1, 32] = 1.000
                elif(discScheme == "Empirical" and device == "0.67_NIR_6BIT"):
                        # Define odd and even column states
                        self.pLevel = np.zeros((2, 65))  # Create a 2x64 array for odd and even column states

                        # Odd column states
                        self.pLevel[0, 0] = 0.000
                        self.pLevel[0, 1] = 0.013
                        self.pLevel[0, 2] = 0.028
                        self.pLevel[0, 3] = 0.063
                        self.pLevel[0, 4] = 0.025
                        self.pLevel[0, 5] = 0.039
                        self.pLevel[0, 6] = 0.057
                        self.pLevel[0, 7] = 0.094
                        self.pLevel[0, 8] = 0.138
                        self.pLevel[0, 9] = 0.176
                        self.pLevel[0, 10] = 0.210
                        self.pLevel[0, 11] = 0.275
                        self.pLevel[0, 12] = 0.195
                        self.pLevel[0, 13] = 0.235
                        self.pLevel[0, 14] = 0.272
                        self.pLevel[0, 15] = 0.337
                        self.pLevel[0, 16] = 0.374
                        self.pLevel[0, 17] = 0.406
                        self.pLevel[0, 18] = 0.449
                        self.pLevel[0, 19] = 0.508
                        self.pLevel[0, 20] = 0.430
                        self.pLevel[0, 21] = 0.463
                        self.pLevel[0, 22] = 0.510
                        self.pLevel[0, 23] = 0.573
                        self.pLevel[0, 24] = 0.614
                        self.pLevel[0, 25] = 0.668
                        self.pLevel[0, 26] = 0.738
                        self.pLevel[0, 27] = 0.822
                        self.pLevel[0, 28] = 0.704
                        self.pLevel[0, 29] = 0.757
                        self.pLevel[0, 30] = 0.830
                        self.pLevel[0, 31] = 0.918
                        self.pLevel[0, 32] = 0.025
                        self.pLevel[0, 33] = 0.039
                        self.pLevel[0, 34] = 0.053
                        self.pLevel[0, 35] = 0.089
                        self.pLevel[0, 36] = 0.050
                        self.pLevel[0, 37] = 0.065
                        self.pLevel[0, 38] = 0.081
                        self.pLevel[0, 39] = 0.120
                        self.pLevel[0, 40] = 0.194
                        self.pLevel[0, 41] = 0.233
                        self.pLevel[0, 42] = 0.264
                        self.pLevel[0, 43] = 0.328
                        self.pLevel[0, 44] = 0.249
                        self.pLevel[0, 45] = 0.289
                        self.pLevel[0, 46] = 0.322
                        self.pLevel[0, 47] = 0.388
                        self.pLevel[0, 48] = 0.438
                        self.pLevel[0, 49] = 0.469
                        self.pLevel[0, 50] = 0.511
                        self.pLevel[0, 51] = 0.570
                        self.pLevel[0, 52] = 0.489
                        self.pLevel[0, 53] = 0.522
                        self.pLevel[0, 54] = 0.566
                        self.pLevel[0, 55] = 0.629
                        self.pLevel[0, 56] = 0.704
                        self.pLevel[0, 57] = 0.753
                        self.pLevel[0, 58] = 0.819
                        self.pLevel[0, 59] = 0.901
                        self.pLevel[0, 60] = 0.783
                        self.pLevel[0, 61] = 0.832
                        self.pLevel[0, 62] = 0.899
                        self.pLevel[0, 63] = 0.984
                        self.pLevel[0, 64] = 1.000

                        # Even column states
                        self.pLevel[1, 0] = 0.007
                        self.pLevel[1, 1] = 0.016
                        self.pLevel[1, 2] = 0.029
                        self.pLevel[1, 3] = 0.059
                        self.pLevel[1, 4] = 0.031
                        self.pLevel[1, 5] = 0.041
                        self.pLevel[1, 6] = 0.056
                        self.pLevel[1, 7] = 0.089
                        self.pLevel[1, 8] = 0.138
                        self.pLevel[1, 9] = 0.172
                        self.pLevel[1, 10] = 0.203
                        self.pLevel[1, 11] = 0.262
                        self.pLevel[1, 12] = 0.193
                        self.pLevel[1, 13] = 0.228
                        self.pLevel[1, 14] = 0.261
                        self.pLevel[1, 15] = 0.322
                        self.pLevel[1, 16] = 0.389
                        self.pLevel[1, 17] = 0.415
                        self.pLevel[1, 18] = 0.457
                        self.pLevel[1, 19] = 0.511
                        self.pLevel[1, 20] = 0.438
                        self.pLevel[1, 21] = 0.466
                        self.pLevel[1, 22] = 0.510
                        self.pLevel[1, 23] = 0.565
                        self.pLevel[1, 24] = 0.629
                        self.pLevel[1, 25] = 0.674
                        self.pLevel[1, 26] = 0.739
                        self.pLevel[1, 27] = 0.815
                        self.pLevel[1, 28] = 0.696
                        self.pLevel[1, 29] = 0.744
                        self.pLevel[1, 30] = 0.812
                        self.pLevel[1, 31] = 0.884
                        self.pLevel[1, 32] = 0.035
                        self.pLevel[1, 33] = 0.045
                        self.pLevel[1, 34] = 0.056
                        self.pLevel[1, 35] = 0.088
                        self.pLevel[1, 36] = 0.059
                        self.pLevel[1, 37] = 0.070
                        self.pLevel[1, 38] = 0.083
                        self.pLevel[1, 39] = 0.117
                        self.pLevel[1, 40] = 0.199
                        self.pLevel[1, 41] = 0.233
                        self.pLevel[1, 42] = 0.261
                        self.pLevel[1, 43] = 0.320
                        self.pLevel[1, 44] = 0.251
                        self.pLevel[1, 45] = 0.286
                        self.pLevel[1, 46] = 0.316
                        self.pLevel[1, 47] = 0.376
                        self.pLevel[1, 48] = 0.469
                        self.pLevel[1, 49] = 0.495
                        self.pLevel[1, 50] = 0.535
                        self.pLevel[1, 51] = 0.589
                        self.pLevel[1, 52] = 0.512
                        self.pLevel[1, 53] = 0.539
                        self.pLevel[1, 54] = 0.581
                        self.pLevel[1, 55] = 0.637
                        self.pLevel[1, 56] = 0.757
                        self.pLevel[1, 57] = 0.795
                        self.pLevel[1, 58] = 0.860
                        self.pLevel[1, 59] = 0.930
                        self.pLevel[1, 60] = 0.800
                        self.pLevel[1, 61] = 0.837
                        self.pLevel[1, 62] = 0.900
                        self.pLevel[1, 63] = 0.968
                        self.pLevel[1, 64] = 1.000
                elif(discScheme =="Empirical" and device == "0.47_SHRINK"):
                        self.pLevel = np.zeros((5,))
                        self.pLevel[0] = 0
                        self.pLevel[1] = 0.01
                        self.pLevel[2] = 0.20
                        self.pLevel[3] = 0.75
                        self.pLevel[4] = 1
                elif(discScheme =="Linear" and (device == "0.47" or device == "0.67" or device == "0.98")):
                        self.pLevel = np.linspace(0,1,self.nLevel+1)
                elif(discScheme =="Linear" and (device == "0.67_NIR_5BIT" or device == "0.67_NIR_6BIT")):
                        self.pLevel = np.zeros((2,self.nLevel))
                        self.pLevel[0,:] = np.linspace(0,1,self.nLevel)
                # Modified the phase levels if the mirror bias (lambda_global) is not the same as lambda_m.
                # This updates the state-to-phase table as well as the phase wrapping.
                if(lambdaGlobal != None and lambdaGlobal > 0 and not (discScheme == "Empirical" and (device == "0.47" or device == "0.67" or device == "0.98"))):
                        self.pLevel = self.pLevel * lambdaGlobal / lambdaM
                        self.pLevel = np.remainder(self.pLevel,1)

                # Define memory layouts and evm mapping compatibility parameters
                if(device == "0.47"):
                        self.memory_layout = [[2, 3],[0, 1]]
                        self.memory_lut = [3, 2, 1, 7, 0, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12]
                        self.evm_padding = 0
                        self.evm_fliplr = False
                        self.evm_flipud = False
                elif(device == "0.67"):
                        self.memory_layout = [[1, 3],[0, 2]]
                        self.memory_lut = [3, 2, 1, 7, 0, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12]
                        self.evm_padding = 0
                        self.evm_fliplr = True
                        self.evm_flipud = False
                elif(device == "0.98"):
                        self.memory_layout = [[0, 1],[2, 3]]
                        self.memory_lut = [3, 2, 1, 7, 0, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12]
                        self.evm_padding = 0
                        self.evm_fliplr = False
                        self.evm_flipud = False
                elif(device == "0.67_NIR_5BIT"):
                        self.memory_layout = [[3, 1, 2],[2, 0, 4]]
                        self.memory_lut = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                           17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
                        self.evm_padding = 2
                        self.evm_fliplr = True
                        self.evm_flipud = False
                elif(device == "0.67_NIR_6BIT"):
                        self.memory_layout = [[3, 1, 5],[2, 0, 4]]
                        self.memory_lut = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                           17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                           32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 
                                           47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 
                                           62, 63]
                        self.evm_padding = 2
                        self.evm_fliplr = True
                        self.evm_flipud = False
                elif(device == "0.47_SHRINK"):
                        self.memory_layout = None
                        self.evm_padding = None
                        self.evm_fliplr = None
                        self.evm_flipud = None


                # Returns a dictionary that describes the device and its mapping
                DeviceDict = {"device": device,
                              "discScheme": self.discScheme,
                              "lutWavelength": lutWavelength,
                              "pitchW": self.pitchW,
                              "pitchH": self.pitchH,
                              "w": self.w,
                              "h": self.h,
                              "nLevel": self.nLevel,
                              "pLevel": self.pLevel,
                              "memory_layout": self.memory_layout,
                              "memory_lut": self.memory_lut,
                              "evm_padding": self.evm_padding,
                              "evm_fliplr": self.evm_fliplr,
                              "evm_flipud": self.evm_flipud}
                
                return DeviceDict
        
################################
##  TI PLM Mapping functions  ##
################################

        # General function for generating PLM output
        def formatPLM(self, DeviceDictionary, CGH_state_map):
                CGH_mapped = None
                device = DeviceDictionary["device"]
                nLevel = DeviceDictionary["nLevel"]
                memoryLayout = DeviceDictionary["memory_layout"]
                memoryLut = DeviceDictionary["memory_lut"]
                padding = DeviceDictionary["evm_padding"]
                evmFlipLR = DeviceDictionary["evm_fliplr"]
                evmFlipUD = DeviceDictionary["evm_flipud"]
                if(memoryLayout is None):
                        if 'ti_phase_light_modulator' in sys.modules:
                                try:
                                        # Initialize PLM object for the device
                                        plmMap = PLM.from_db('p' + device.split('.')[-1])

                                        # Compute bmp image for the given phase map
                                        CGH_mapped = plmMap.electrode_map(CGH_state_map.astype(np.uint8)).astype(np.float64)
                                except Exception:
                                        CGH_mapped = None
                        if(CGH_mapped is None and device == "0.47"):
                                CGH_mapped = self.formatPLM_p47(CGH_state_map, nLevel)
                        elif(CGH_mapped is None and device == "0.67"):
                                CGH_mapped = self.formatPLM_p67(CGH_state_map, nLevel)
                        elif(CGH_mapped is None and device == "0.98"):
                                CGH_mapped = self.formatPLM_p98(CGH_state_map, nLevel)
                        elif(CGH_mapped is None and device == "0.67_NIR_5BIT"):
                                CGH_mapped = self.formatPLM_Generic(CGH_state_map, memoryLayout, padding=2,fliplr=True)
                        elif(CGH_mapped is None and device == "0.67_NIR_6BIT"):
                                CGH_mapped = self.formatPLM_Generic(CGH_state_map, memoryLayout, padding=2,fliplr=True)
                        elif(CGH_mapped is None and device == "0.47_SHRINK"):
                                CGH_mapped = self.formatPLM_p47_Shrink(CGH_state_map, nLevel)
                else:
                        CGH_mapped = self.formatPLM_Generic(CGH_state_map, memoryLayout, memoryLut, padding, evmFlipLR, evmFlipUD)
                return CGH_mapped

        def formatPLMStack(self, DeviceDictionary, CGH_state_stack):
                state_stack = np.asarray(CGH_state_stack)
                if(state_stack.ndim != 3):
                        raise ValueError("CGH_state_stack must have shape [frames, height, width].")

                device = DeviceDictionary["device"]
                nLevel = DeviceDictionary["nLevel"]
                memoryLayout = DeviceDictionary["memory_layout"]
                memoryLut = DeviceDictionary["memory_lut"]
                padding = DeviceDictionary["evm_padding"]
                evmFlipLR = DeviceDictionary["evm_fliplr"]
                evmFlipUD = DeviceDictionary["evm_flipud"]
                standardP67EVMLayout = (
                        memoryLayout is not None
                        and np.array_equal(np.asarray(memoryLayout), np.asarray([[1, 3], [0, 2]]))
                        and memoryLut is not None
                        and list(memoryLut) == [3, 2, 1, 7, 0, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12]
                        and padding == 0
                        and bool(evmFlipLR)
                        and not bool(evmFlipUD)
                )
                if(device == "0.67" and (memoryLayout is None or standardP67EVMLayout)):
                        return self.formatPLM_p67_stack(state_stack, nLevel)

                mapped_planes = []
                for frame_idx in range(state_stack.shape[0]):
                        mapped_planes.append(self.formatPLM(DeviceDictionary, state_stack[frame_idx]))
                return np.stack(mapped_planes, axis=2)

        # Format discretized final phase hologram for 0.47 PLM output
        def formatPLM_p47(self, CGH_phase_map, nLevel):
                pslm_stateq = CGH_phase_map

                # Initialize electrode_map with NaN
                electrode_map = np.full((2, 2, nLevel), np.nan)

                # Populate electrode_map
                for i in range(nLevel):  # Python uses 0-based indexing
                        if i == 0:
                                electrode_map[0, 1, i] = 0  # E3
                                electrode_map[0, 0, i] = 0  # E2
                                electrode_map[1, 1, i] = 1  # E1
                                electrode_map[1, 0, i] = 1  # E0
                        elif i == 1:
                                electrode_map[0, 1, i] = 0  # E3
                                electrode_map[0, 0, i] = 0  # E2
                                electrode_map[1, 1, i] = 1  # E1
                                electrode_map[1, 0, i] = 0  # E0
                        elif i == 2:
                                electrode_map[0, 1, i] = 0  # E3
                                electrode_map[0, 0, i] = 0  # E2
                                electrode_map[1, 1, i] = 0  # E1
                                electrode_map[1, 0, i] = 1  # E0
                        elif i == 3:
                                electrode_map[0, 1, i] = 0  # E3
                                electrode_map[0, 0, i] = 1  # E2
                                electrode_map[1, 1, i] = 1  # E1
                                electrode_map[1, 0, i] = 1  # E0
                        elif i == 4:
                                electrode_map[0, 1, i] = 0  # E3
                                electrode_map[0, 0, i] = 0  # E2
                                electrode_map[1, 1, i] = 0  # E1
                                electrode_map[1, 0, i] = 0  # E0
                        elif i == 5:
                                electrode_map[0, 1, i] = 0  # E3
                                electrode_map[0, 0, i] = 1  # E2
                                electrode_map[1, 1, i] = 1  # E1
                                electrode_map[1, 0, i] = 0  # E0
                        elif i == 6:
                                electrode_map[0, 1, i] = 0  # E3
                                electrode_map[0, 0, i] = 1  # E2
                                electrode_map[1, 1, i] = 0  # E1
                                electrode_map[1, 0, i] = 1  # E0
                        elif i == 7:
                                electrode_map[0, 1, i] = 0  # E3
                                electrode_map[0, 0, i] = 1  # E2
                                electrode_map[1, 1, i] = 0  # E1
                                electrode_map[1, 0, i] = 0  # E0
                        elif i == 8:
                                electrode_map[0, 1, i] = 1  # E3
                                electrode_map[0, 0, i] = 0  # E2
                                electrode_map[1, 1, i] = 1  # E1
                                electrode_map[1, 0, i] = 1  # E0
                        elif i == 9:
                                electrode_map[0, 1, i] = 1  # E3
                                electrode_map[0, 0, i] = 0  # E2
                                electrode_map[1, 1, i] = 1  # E1
                                electrode_map[1, 0, i] = 0  # E0
                        elif i == 10:
                                electrode_map[0, 1, i] = 1  # E3
                                electrode_map[0, 0, i] = 0  # E2
                                electrode_map[1, 1, i] = 0  # E1
                                electrode_map[1, 0, i] = 1  # E0
                        elif i == 11:
                                electrode_map[0, 1, i] = 1  # E3
                                electrode_map[0, 0, i] = 0  # E2
                                electrode_map[1, 1, i] = 0  # E1
                                electrode_map[1, 0, i] = 0  # E0
                        elif i == 12:
                                electrode_map[0, 1, i] = 1  # E3
                                electrode_map[0, 0, i] = 1  # E2
                                electrode_map[1, 1, i] = 1  # E1
                                electrode_map[1, 0, i] = 1  # E0
                        elif i == 13:
                                electrode_map[0, 1, i] = 1  # E3
                                electrode_map[0, 0, i] = 1  # E2
                                electrode_map[1, 1, i] = 1  # E1
                                electrode_map[1, 0, i] = 0  # E0
                        elif i == 14:
                                electrode_map[0, 1, i] = 1  # E3
                                electrode_map[0, 0, i] = 1  # E2
                                electrode_map[1, 1, i] = 0  # E1
                                electrode_map[1, 0, i] = 1  # E0
                        elif i == 15:
                                electrode_map[0, 1, i] = 1  # E3
                                electrode_map[0, 0, i] = 1  # E2
                                electrode_map[1, 1, i] = 0  # E1
                                electrode_map[1, 0, i] = 0  # E0

                # Get the size of pslm_stateq
                rows, cols = pslm_stateq.shape

                # Initialize pslm_logical
                pslm_logical = np.zeros((rows * 2, cols * 2))

                # Populate pslm_logical
                for row in range(rows):
                        for col in range(cols):
                                state = int(pslm_stateq[row, col])  # Get the state and cast to avoid error
                                pslm_logical[2 * row, 2 * col + 1] = electrode_map[0, 1, state]  # E3
                                pslm_logical[2 * row, 2 * col] = electrode_map[0, 0, state]      # E2
                                pslm_logical[2 * row + 1, 2 * col + 1] = electrode_map[1, 1, state]  # E1
                                pslm_logical[2 * row + 1, 2 * col] = electrode_map[1, 0, state]      # E0

                return pslm_logical * 255

        @staticmethod
        def _p67_electrode_bits():
                return np.array([
                        [0, 0, 1, 1],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [0, 1, 1, 1],
                        [0, 0, 0, 0],
                        [0, 1, 1, 0],
                        [0, 1, 0, 1],
                        [0, 1, 0, 0],
                        [1, 0, 1, 1],
                        [1, 0, 1, 0],
                        [1, 0, 0, 1],
                        [1, 0, 0, 0],
                        [1, 1, 1, 1],
                        [1, 1, 1, 0],
                        [1, 1, 0, 1],
                        [1, 1, 0, 0],
                ], dtype=np.uint8)

        def _p67_flipped_tiles(self):
                electrode_bits = self._p67_electrode_bits()
                tiles = np.zeros((16, 2, 2), dtype=np.uint8)
                tiles[:, 0, 1] = electrode_bits[:, 0] * 255  # E3
                tiles[:, 1, 1] = electrode_bits[:, 1] * 255  # E2
                tiles[:, 0, 0] = electrode_bits[:, 2] * 255  # E1
                tiles[:, 1, 0] = electrode_bits[:, 3] * 255  # E0
                return np.flip(tiles, axis=2)

        @staticmethod
        def _state_indices_uint8(CGH_state_map, nLevel):
                stateq = np.asarray(CGH_state_map)
                if(stateq.dtype == np.uint8):
                        if(stateq.size > 0 and np.max(stateq) >= nLevel):
                                return np.minimum(stateq, nLevel - 1).astype(np.uint8, copy=False)
                        return stateq

                return np.clip(stateq, 0, nLevel - 1).astype(np.uint8, copy=False)

        # Format discretized final phase hologram for 0.67 PLM output
        def formatPLM_p67(self, CGH_phase_map, nLevel):
                if(nLevel != 16):
                        raise ValueError("0.67 PLM formatting expects 16 phase levels.")

                stateq = self._state_indices_uint8(CGH_phase_map, nLevel)
                rows, cols = stateq.shape
                mapped_tiles = self._p67_flipped_tiles()[stateq[:, ::-1]]
                return mapped_tiles.transpose(0, 2, 1, 3).reshape(rows * 2, cols * 2)

        def formatPLM_p67_stack(self, CGH_state_stack, nLevel):
                if(nLevel != 16):
                        raise ValueError("0.67 PLM formatting expects 16 phase levels.")

                state_stack = np.asarray(CGH_state_stack)
                if(state_stack.ndim != 3):
                        raise ValueError("CGH_state_stack must have shape [frames, height, width].")

                stateq = self._state_indices_uint8(state_stack, nLevel)
                frames, rows, cols = stateq.shape
                mapped_tiles = self._p67_flipped_tiles()[stateq[:, :, ::-1]]
                mapped_stack = mapped_tiles.transpose(0, 1, 3, 2, 4).reshape(frames, rows * 2, cols * 2)
                return np.ascontiguousarray(np.moveaxis(mapped_stack, 0, 2))

        # Format discretized final phase hologram for 0.98 PLM output
        def formatPLM_p98(self, CGH_phase_map, nLevel):
                pslm_stateq = CGH_phase_map

                # Initialize electrode_map with NaN
                electrode_map = np.full((2, 2, nLevel), np.nan)

                for i in range(electrode_map.shape[2]):
                        if i == 0:
                                electrode_map[1, 1, i] = 0   #E3
                                electrode_map[1, 0, i] = 0   #E2
                                electrode_map[0, 1, i] = 0   #E1
                                electrode_map[0, 0, i] = 0   #E0
                        elif i == 1:
                                electrode_map[1, 1, i] = 0   #E3
                                electrode_map[1, 0, i] = 0   #E2
                                electrode_map[0, 1, i] = 0   #E1
                                electrode_map[0, 0, i] = 1   #E0
                        elif i == 2:
                                electrode_map[1, 1, i] = 0   #E3
                                electrode_map[1, 0, i] = 0   #E2
                                electrode_map[0, 1, i] = 1   #E1
                                electrode_map[0, 0, i] = 0   #E0
                        elif i == 3:
                                electrode_map[1, 1, i] = 0   #E3
                                electrode_map[1, 0, i] = 1   #E2
                                electrode_map[0, 1, i] = 0   #E1
                                electrode_map[0, 0, i] = 0   #E0
                        elif i == 4:
                                electrode_map[1, 1, i] = 0   #E3
                                electrode_map[1, 0, i] = 0   #E2
                                electrode_map[0, 1, i] = 1   #E1
                                electrode_map[0, 0, i] = 1   #E0
                        elif i == 5:
                                electrode_map[1, 1, i] = 0   #E3
                                electrode_map[1, 0, i] = 1   #E2
                                electrode_map[0, 1, i] = 0   #E1
                                electrode_map[0, 0, i] = 1   #E0
                        elif i == 6:
                                electrode_map[1, 1, i] = 0   #E3
                                electrode_map[1, 0, i] = 1   #E2
                                electrode_map[0, 1, i] = 1   #E1
                                electrode_map[0, 0, i] = 0   #E0
                        elif i == 7:
                                electrode_map[1, 1, i] = 0   #E3
                                electrode_map[1, 0, i] = 1   #E2
                                electrode_map[0, 1, i] = 1   #E1
                                electrode_map[0, 0, i] = 1   #E0
                        elif i == 8:
                                electrode_map[1, 1, i] = 1   #E3
                                electrode_map[1, 0, i] = 0   #E2
                                electrode_map[0, 1, i] = 0   #E1
                                electrode_map[0, 0, i] = 0   #E0
                        elif i == 9:
                                electrode_map[1, 1, i] = 1   #E3
                                electrode_map[1, 0, i] = 0   #E2
                                electrode_map[0, 1, i] = 0   #E1
                                electrode_map[0, 0, i] = 1   #E0
                        elif i == 10:
                                electrode_map[1, 1, i] = 1   #E3
                                electrode_map[1, 0, i] = 0   #E2
                                electrode_map[0, 1, i] = 1   #E1
                                electrode_map[0, 0, i] = 0   #E0
                        elif i == 11:
                                electrode_map[1, 1, i] = 1   #E3
                                electrode_map[1, 0, i] = 0   #E2
                                electrode_map[0, 1, i] = 1   #E1
                                electrode_map[0, 0, i] = 1   #E0
                        elif i == 12:
                                electrode_map[1, 1, i] = 1   #E3
                                electrode_map[1, 0, i] = 1   #E2
                                electrode_map[0, 1, i] = 0   #E1
                                electrode_map[0, 0, i] = 0   #E0
                        elif i == 13:
                                electrode_map[1, 1, i] = 1   #E3
                                electrode_map[1, 0, i] = 1   #E2
                                electrode_map[0, 1, i] = 0   #E1
                                electrode_map[0, 0, i] = 1   #E0
                        elif i == 14:
                                electrode_map[1, 1, i] = 1   #E3
                                electrode_map[1, 0, i] = 1   #E2
                                electrode_map[0, 1, i] = 1   #E1
                                electrode_map[0, 0, i] = 0   #E0
                        elif i == 15:
                                electrode_map[1, 1, i] = 1   #E3
                                electrode_map[1, 0, i] = 1   #E2
                                electrode_map[0, 1, i] = 1   #E1
                                electrode_map[0, 0, i] = 1   #E0

                # Get the size of pslm_stateq
                rows, cols = pslm_stateq.shape

                # Initialize pslm_logical
                pslm_logical = np.zeros((rows * 2, cols * 2))

                # Populate pslm_logical
                for row in range(rows):
                        for col in range(cols):
                                state = int(pslm_stateq[row, col])  # Get the state and cast to avoid error
                                pslm_logical[2 * row, 2 * col + 1] = electrode_map[0, 1, state]  # E3
                                pslm_logical[2 * row, 2 * col] = electrode_map[0, 0, state]      # E2
                                pslm_logical[2 * row + 1, 2 * col + 1] = electrode_map[1, 1, state]  # E1
                                pslm_logical[2 * row + 1, 2 * col] = electrode_map[1, 0, state]      # E0


                return pslm_logical * 255

        # Format discretized final phsae hologram for .47 PLM shrink 
        def formatPLM_p47_Shrink(self, CGH_phase_map, nLevel):
                pslm_stateq = CGH_phase_map

                # Initialize electrode_map with NaN
                electrode_map = np.full((2, 2, nLevel), np.nan)

                for i in range(nLevel):
                        if i == 0:
                                electrode_map[0, 1, i] = 0  # E
                                electrode_map[1, 0, i] = 1  # E
                        elif i == 1:
                                electrode_map[0, 1, i] = 1  # E
                                electrode_map[1, 0, i] = 1  # E
                        elif i == 2:
                                electrode_map[0, 1, i] = 0  # E
                                electrode_map[1, 0, i] = 0  # E
                        elif i == 3:
                                electrode_map[0, 1, i] = 1  # E
                                electrode_map[1, 0, i] = 0  # E

                rows, cols = pslm_stateq.shape
                pslm_logical = np.zeros((rows * 2, cols))
                for row in range(rows):
                        for col in range(cols):
                                state = int(pslm_stateq[row, col])  # Get the state and cast to avoid error
                                pslm_logical[2 * row, col] = electrode_map[0, 1, state]  # E
                                pslm_logical[2 * row + 1, col] = electrode_map[1, 0, state]  # E
        
                return pslm_logical * 255

        # Format all PLMs using the memory layout, phase map, and padding provided
        def formatPLM_Generic(self, CGH_phase_map, memory_layout, memory_lut, padding = None, fliplr = False, flipud = False):
                pslm_stateq = CGH_phase_map.astype(int)
                height = pslm_stateq.shape[0]
                width = pslm_stateq.shape[1]
                memX = len(memory_layout[0])
                memY = len(memory_layout)

                pslm_logical = np.zeros((pslm_stateq.shape[0]*len(memory_layout), pslm_stateq.shape[1]*len(memory_layout[0])), dtype=int)

                for y in range(height):
                        for x in range(width):
                                memVal = memory_lut[pslm_stateq[y, x]]
                                for m in range(len(memory_layout)):
                                        for n in range(len(memory_layout[0])):
                                                bit_position = memory_layout[m][n]
                                                bit_value = (memVal >> bit_position) & 1
                                                pslm_logical[y*memY + m, x*memX + n] = bit_value

                # Pad the padding value in columns on left and right
                if padding != None and padding != 0:
                        pslm_logical = np.pad(pslm_logical, ((0, 0), (padding, padding)), 'constant', constant_values=0)

                # Flip image horizontally
                if fliplr:
                        pslm_logical = np.fliplr(pslm_logical.astype(bool))
                
                # Flip image vertically
                if flipud:
                        pslm_logical = np.flipud(pslm_logical.astype(bool))

                # Store encoded result (scaled by 255)
                return (pslm_logical.astype(np.uint8) * 255) 

class CGHGenerator():

        # Final CGH that has been mapped to the TI PLM electrodes
        CGH_mapped = None
        # Final CGH Hologram. Discretized but not mapped to TI PLM electrodes
        CGH_phase = None

        # Final Recovered CGH image
        imRecovered_cont = None
        imRecovered_disc = None
        imRecovered_cont_native = None
        imRecovered_disc_native = None

        # Target Illumination for display
        targetIllumination = None
        optimizedIllumination = None
        illuminationWasOptimized = False

        # Raw CGH image in a continuous and discrete form
        CGH_output_cont = None
        CGH_output_cont_subframes = None
        temporalFrameLUTWavelengths = None
        temporalFrameIlluminationWavelengths = None
        CGH_output_phase_disc = None # Final, discretized phase at hologram plane
        CGH_output_phase_disc_subframes = None
        CGH_output_state_disc = None # Final, discretized states used for CGH
        CGH_output_state_disc_subframes = None
        CGH_effective_field_cont = None
        CGH_effective_field_disc = None
        CGH_effective_field_cont_subframes = None
        CGH_effective_field_disc_subframes = None
        CGH_is_sequence = False
        softWTAGroupLUTWavelengths = None
        softWTAGroupTemporalFrames = None
        softWTASelectorWeights = None
        softWTAHardWinnerMap = None
        softWTASoftFusedNative = None
        softWTAHardFusedNative = None
        softWTAChosenFusionMode = None
        softWTAChosenFusionMSE = None
        softWTAChosenFusionPSNR = None
        softWTAGroupReconsNative = None
        softWTAGroupOutputs = None

        # Custom seed phase used for CGH generation
        seedPhase = None

        # Target image
        imTarget = None

        # Object of DeviceLibrary
        deviceLibary = DeviceLibrary()

        # Wavelength in meters
        lambda_m = 532e-9
        prop_dist = None

        # Number of bitplanes in the CGH
        bitPlanes = 1

        # input to show image in a window
        show_images = False
        lossMode = "auto"

        # input to load algorithms to GPU or CPU
        MLDevice = torch.device("cuda" if torch.cuda.is_available() else "cpu") if torch is not None else "cpu"

        @staticmethod
        def requireTorch(featureName = "This feature"):
                if(torch is None):
                        raise ImportError(f"{featureName} requires PyTorch. {_TORCH_INSTALL_HINT}")

        @staticmethod
        def normalizePixelGrouping(pixelGrouping = 1):
                if(isinstance(pixelGrouping, (int, np.integer))):
                        group_h = int(pixelGrouping)
                        group_w = int(pixelGrouping)
                elif(isinstance(pixelGrouping, (tuple, list)) and len(pixelGrouping) == 2):
                        group_h = int(pixelGrouping[0])
                        group_w = int(pixelGrouping[1])
                else:
                        raise ValueError("pixelGrouping must be an int or a (height, width) tuple.")

                if(group_h < 1 or group_w < 1):
                        raise ValueError("pixelGrouping values must be >= 1.")

                return group_h, group_w

        @staticmethod
        def expandGroupedImage(image, pixelGrouping = 1):
                group_h, group_w = CGHGenerator.normalizePixelGrouping(pixelGrouping)

                if(group_h == 1 and group_w == 1):
                        return image.copy()

                return np.repeat(np.repeat(image, group_h, axis=0), group_w, axis=1)

        def resolveInitialPhaseMap(self, initialPhase = "Random", allowTemporalStack = False):
                phase_shape = (self.imTarget.shape[0], self.imTarget.shape[1])
                initialPhaseMode = str(initialPhase).upper()

                if(initialPhaseMode == "RANDOM"):
                        return (2*np.pi * np.random.rand(*phase_shape)).astype(np.float32)
                if(initialPhaseMode == "UNITY"):
                        return np.full(phase_shape, 2*np.pi, dtype=np.float32)
                if(initialPhaseMode == "CUSTOM"):
                        if(self.seedPhase is None):
                                print("Error: No seed phase defined. Using random seed")
                                return (2*np.pi * np.random.rand(*phase_shape)).astype(np.float32)

                        seed_phase = np.asarray(self.seedPhase, dtype=np.float32)
                        if(seed_phase.shape != phase_shape):
                                if(allowTemporalStack and seed_phase.ndim == 3 and seed_phase.shape[1:] == phase_shape):
                                        return seed_phase[0].copy()
                                raise ValueError("Error: Seed phase must be same size as target image")
                        return seed_phase.copy()

                print("Error: Invalid initial phase seed. Using random seed")
                return (2*np.pi * np.random.rand(*phase_shape)).astype(np.float32)

        def buildTemporalInitialPhaseStack(self, temporalFrames, basePhase, initialPhase = "Random", temporalSeedJitter = 0.10):
                temporalFrames = int(max(1, temporalFrames))
                phase_shape = (self.imTarget.shape[0], self.imTarget.shape[1])

                phase_stack = None
                use_exact_stack = False

                if(initialPhase.upper() == "CUSTOM" and self.seedPhase is not None):
                        seed_phase = np.asarray(self.seedPhase, dtype=np.float32)
                        if(seed_phase.shape == (temporalFrames,) + phase_shape):
                                phase_stack = np.mod(seed_phase.copy(), 2*np.pi)
                                use_exact_stack = True
                        elif(seed_phase.shape == phase_shape):
                                phase_stack = np.repeat(seed_phase[None, :, :], temporalFrames, axis=0)
                        else:
                                raise ValueError(
                                        "Custom temporal seedPhase must have shape "
                                        f"{phase_shape} or {(temporalFrames,) + phase_shape}, "
                                        f"got {seed_phase.shape}."
                                )

                if(phase_stack is None):
                        base_phase = np.asarray(basePhase, dtype=np.float32)
                        if(base_phase.shape == ()):
                                base_phase = np.full(phase_shape, float(base_phase), dtype=np.float32)
                        else:
                                base_phase = np.broadcast_to(base_phase, phase_shape).astype(np.float32, copy=False)
                        phase_stack = np.repeat(base_phase[None, :, :], temporalFrames, axis=0)

                if(temporalFrames == 1):
                        return np.mod(phase_stack[:1], 2*np.pi).astype(np.float32).copy()

                if((not use_exact_stack) and temporalSeedJitter is not None and float(temporalSeedJitter) > 0):
                        jitter = np.random.randn(*phase_stack.shape).astype(np.float32) * float(temporalSeedJitter)
                        jitter[0, :, :] = 0.0
                        phase_stack = np.mod(phase_stack + jitter, 2*np.pi)

                return np.mod(phase_stack, 2*np.pi).astype(np.float32)

        @staticmethod
        def fieldToIntensity(field):
                return np.abs(field)**2

        @staticmethod
        def scaleIntensityToTarget(intensity, target_intensity, eps = 1e-10):
                scale = np.mean(intensity*target_intensity) / (np.mean(intensity**2) + eps)
                return scale, scale*intensity

        @staticmethod
        def torchFieldToIntensity(field):
                return torch.abs(field)**2

        @staticmethod
        def torchScaleIntensityToTarget(intensity, target_intensity, eps = 1e-10):
                scale = (intensity*target_intensity).mean(dim=(-1,-2), keepdims=True) / (
                        (intensity**2).mean(dim=(-1,-2), keepdims=True) + eps
                )
                return scale, scale*intensity

        def _targetIsBinary(self):
                if(self.imTarget is None):
                        return False
                return np.all((self.imTarget == 0) | (self.imTarget == 1))

        def _resolveLossMode(self, lossMode):
                mode = str(lossMode).replace("-", "_").replace(" ", "_").lower()
                if(mode == "auto"):
                        return "balanced" if self._targetIsBinary() else "mse"
                if(mode not in ("mse", "balanced")):
                        raise ValueError("lossMode must be 'auto', 'mse', or 'balanced'.")
                return mode

        @staticmethod
        def _torchLossWeights(target, lossMode, eps=1e-8):
                if(lossMode != "balanced"):
                        return None

                foreground = target > 0.5
                background = ~foreground
                fg_fraction = torch.clamp(foreground.float().mean(), min=eps)
                bg_fraction = torch.clamp(background.float().mean(), min=eps)

                if(bool((foreground.any() & background.any()).detach().cpu())):
                        weights = torch.where(foreground, 0.5 / fg_fraction, 0.5 / bg_fraction)
                        return weights / torch.clamp(weights.mean(), min=eps)

                return None

        @staticmethod
        def _torchWeightedGain(prediction, target, weights=None, eps=1e-20):
                if(weights is None):
                        denom = torch.clamp((prediction ** 2).mean(dim=(-1, -2), keepdim=True), min=eps)
                        return (prediction * target).mean(dim=(-1, -2), keepdim=True) / denom

                denom = torch.clamp((weights * prediction ** 2).mean(dim=(-1, -2), keepdim=True), min=eps)
                return (weights * prediction * target).mean(dim=(-1, -2), keepdim=True) / denom

        @staticmethod
        def _torchWeightedMSE(prediction, target, weights=None):
                error = (prediction - target) ** 2
                if(weights is not None):
                        error = weights * error
                return error.mean()

        @staticmethod
        def normalizeIlluminationProfile(profile, eps = 1e-6, normalizeMean = True):
                profile = np.asarray(profile, dtype=np.float64)
                if(profile.ndim != 2):
                        raise ValueError("Illumination profiles must be 2D arrays.")

                profile = np.maximum(profile, float(eps))

                if(normalizeMean):
                        mean_value = float(np.mean(profile))
                        if(mean_value <= float(eps)):
                                raise ValueError("Illumination profiles must have a positive mean value.")
                        profile = profile / mean_value

                return profile

        @staticmethod
        def torchIlluminationSmoothnessLoss(illumination):
                loss_val = torch.tensor(0.0, dtype=illumination.dtype, device=illumination.device)
                if(illumination.shape[1] > 1):
                        loss_val = loss_val + ((illumination[:, 1:] - illumination[:, :-1])**2).mean()
                if(illumination.shape[0] > 1):
                        loss_val = loss_val + ((illumination[1:, :] - illumination[:-1, :])**2).mean()
                return loss_val

        def _buildIlluminationProfile(self, illumType = "UNIFORM", ill_x0 = 0, ill_y0 = 0,
                                      ill_sigma_x = 1, ill_sigma_y = 1, illuminationProfile = None,
                                      randomStd = 0.25, normalizeMean = False, eps = 1e-6):
                if(self.imTarget is None):
                        raise ValueError("Illumination profiles require imTarget to be defined first.")

                target_shape = (self.imTarget.shape[0], self.imTarget.shape[1])
                illumType = str("UNIFORM" if illumType is None else illumType).upper()

                if(illumType == "UNIFORM"):
                        profile = np.ones(target_shape, dtype=np.float64)
                elif(illumType == "OTHER" or illumType == "GAUSSIAN"):
                        A = 1.0
                        x = np.linspace(1, target_shape[1], target_shape[1])
                        y = np.linspace(1, target_shape[0], target_shape[0])
                        X, Y = np.meshgrid(x, y)
                        res = (((X-float(ill_x0))**2)/(2*float(ill_sigma_x)**2) + ((Y-float(ill_y0))**2)/(2*float(ill_sigma_y)**2))
                        profile = A*np.exp(-res)
                elif(illumType == "CUSTOM" or illumType == "PROFILE"):
                        if(illuminationProfile is None):
                                raise ValueError("CUSTOM illumination requires illuminationProfile to be provided.")
                        profile = np.asarray(illuminationProfile, dtype=np.float64)
                        if(profile.shape != target_shape):
                                raise ValueError(
                                        "Custom illumination profiles must match the target shape. "
                                        f"Expected {target_shape}, got {profile.shape}."
                                )
                elif(illumType == "RANDOM"):
                        profile = np.exp(float(randomStd) * np.random.randn(*target_shape))
                else:
                        raise ValueError(
                                "Unsupported illumination type. Use UNIFORM, OTHER/GAUSSIAN, CUSTOM/PROFILE, or RANDOM."
                        )

                return CGHGenerator.normalizeIlluminationProfile(profile, eps=eps, normalizeMean=normalizeMean)

        def configureIlluminationFromParams(self, algParams = None):
                if(algParams is None):
                        algParams = {}

                illumType = algParams.get("illuminationType", algParams.get("illumType", None))
                illumProfile = algParams.get("illuminationProfile", None)
                if(illumProfile is None):
                        illumProfile = algParams.get("initialIlluminationProfile", None)

                illumCenter = algParams.get("illuminationCenter", None)
                illumSigma = algParams.get("illuminationSigma", None)

                ill_x0 = algParams.get("illuminationX0", algParams.get("ill_x0", 0))
                ill_y0 = algParams.get("illuminationY0", algParams.get("ill_y0", 0))
                ill_sigma_x = algParams.get("illuminationSigmaX", algParams.get("ill_sigma_x", 1))
                ill_sigma_y = algParams.get("illuminationSigmaY", algParams.get("ill_sigma_y", 1))

                if(illumCenter is not None):
                        ill_x0 = illumCenter[0]
                        ill_y0 = illumCenter[1]
                if(illumSigma is not None):
                        ill_sigma_x = illumSigma[0]
                        ill_sigma_y = illumSigma[1]

                normalizeMean = bool(algParams.get("normalizeIllumination", False))
                randomStd = float(algParams.get("illuminationRandomStd", 0.25))
                reuseOptimizedIllumination = bool(algParams.get("reuseOptimizedIllumination", False))

                hasExplicitIllumination = (
                        illumType is not None
                        or illumProfile is not None
                        or illumCenter is not None
                        or illumSigma is not None
                        or ("illuminationX0" in algParams)
                        or ("illuminationY0" in algParams)
                        or ("illuminationSigmaX" in algParams)
                        or ("illuminationSigmaY" in algParams)
                        or ("ill_x0" in algParams)
                        or ("ill_y0" in algParams)
                        or ("ill_sigma_x" in algParams)
                        or ("ill_sigma_y" in algParams)
                )

                if(hasExplicitIllumination):
                        self.createIllumination(
                                illumType="CUSTOM" if illumProfile is not None and illumType is None else illumType,
                                ill_x0=ill_x0,
                                ill_y0=ill_y0,
                                ill_sigma_x=ill_sigma_x,
                                ill_sigma_y=ill_sigma_y,
                                illuminationProfile=illumProfile,
                                randomStd=randomStd,
                                normalizeMean=normalizeMean
                        )
                elif(
                        self.targetIllumination is None
                        or np.asarray(self.targetIllumination).shape != self.imTarget.shape
                        or (self.illuminationWasOptimized and not reuseOptimizedIllumination)
                ):
                        self.createIllumination()
                else:
                        self.targetIllumination = CGHGenerator.normalizeIlluminationProfile(self.targetIllumination, normalizeMean=normalizeMean)
                        self.optimizedIllumination = self.targetIllumination.copy()
                        self.illuminationWasOptimized = False

                return self.targetIllumination.copy()

        def _matchIlluminationToShape(self, targetShape):
                if(self.targetIllumination is None):
                        return None

                illumination = np.asarray(self.targetIllumination, dtype=np.float64)
                targetShape = tuple(targetShape)
                if(illumination.shape == targetShape):
                        return CGHGenerator.normalizeIlluminationProfile(illumination)

                if(targetShape[0] % illumination.shape[0] == 0 and targetShape[1] % illumination.shape[1] == 0):
                        scale_h = int(targetShape[0] // illumination.shape[0])
                        scale_w = int(targetShape[1] // illumination.shape[1])
                        illumination = np.repeat(np.repeat(illumination, scale_h, axis=0), scale_w, axis=1)
                        return CGHGenerator.normalizeIlluminationProfile(illumination)

                raise ValueError(
                        "Illumination profile shape does not match the current hologram shape. "
                        f"Got illumination shape {illumination.shape} and target shape {targetShape}."
                )

        def _updateEffectiveFieldsForCurrentOutput(self):
                if(self.CGH_output_cont is None or self.CGH_output_phase_disc is None):
                        self.CGH_effective_field_cont = None
                        self.CGH_effective_field_disc = None
                        return

                illumination = self._matchIlluminationToShape(np.asarray(self.CGH_output_cont).shape)
                if(illumination is None):
                        self.CGH_effective_field_cont = None
                        self.CGH_effective_field_disc = None
                        return

                self.CGH_effective_field_cont = illumination * np.exp(1j*np.asarray(self.CGH_output_cont, dtype=np.float64))
                self.CGH_effective_field_disc = illumination * np.exp(1j*np.asarray(self.CGH_output_phase_disc, dtype=np.float64))
                self.optimizedIllumination = illumination.copy()

        def _prepareTorchIlluminationOptimization(self, algParams = None, defaultLearningRate = 0.32):
                if(algParams is None):
                        algParams = {}

                illumination_np = CGHGenerator.normalizeIlluminationProfile(
                        self.configureIlluminationFromParams(algParams),
                        normalizeMean=True
                )
                optimizeIllumination = bool(algParams.get("optimizeIllumination", False))
                illuminationLearningRate = float(algParams.get("illuminationLearningRate", defaultLearningRate))
                smoothnessWeight = float(algParams.get("illuminationSmoothnessWeight", 0.0))

                fixed_tensor = torch.tensor(illumination_np, dtype=torch.float32, device=self.MLDevice)
                if(not optimizeIllumination):
                        return {
                                "optimize": False,
                                "tensor": fixed_tensor,
                                "param": None,
                                "optvars": [],
                                "smoothnessWeight": smoothnessWeight,
                        }

                illumination_param = torch.tensor(
                        np.log(np.maximum(illumination_np, 1e-6)),
                        dtype=torch.float32,
                        device=self.MLDevice
                )
                illumination_param.requires_grad_(True)

                return {
                        "optimize": True,
                        "tensor": None,
                        "param": illumination_param,
                        "optvars": [{'params': illumination_param, 'lr': illuminationLearningRate}],
                        "smoothnessWeight": smoothnessWeight,
                }

        def _torchCurrentIllumination(self, illuminationState):
                if(not illuminationState["optimize"]):
                        return illuminationState["tensor"]

                illumination = torch.exp(illuminationState["param"])
                illumination = illumination / torch.clamp(illumination.mean(), min=1e-6)
                return illumination

        def _finalizeTorchIllumination(self, illuminationState, bestIlluminationParam = None):
                if(illuminationState is None):
                        return None

                if(not illuminationState["optimize"]):
                        illumination_np = np.asarray(illuminationState["tensor"].detach().cpu().numpy(), dtype=np.float64)
                        wasOptimized = False
                else:
                        illumination_param = illuminationState["param"] if bestIlluminationParam is None else bestIlluminationParam
                        illumination_tensor = torch.exp(illumination_param)
                        illumination_tensor = illumination_tensor / torch.clamp(illumination_tensor.mean(), min=1e-6)
                        illumination_np = np.asarray(illumination_tensor.detach().cpu().numpy(), dtype=np.float64)
                        wasOptimized = True

                self.targetIllumination = CGHGenerator.normalizeIlluminationProfile(illumination_np)
                self.optimizedIllumination = self.targetIllumination.copy()
                self.illuminationWasOptimized = wasOptimized
                return self.targetIllumination.copy()

        @staticmethod
        def normalizeTemporalFramesPerGroup(temporalFrames, numGroups, name = "temporalFrames"):
                if(isinstance(temporalFrames, (int, np.integer))):
                        values = np.full((int(numGroups),), int(temporalFrames), dtype=np.int64)
                else:
                        values = np.asarray(temporalFrames, dtype=np.int64).reshape(-1)
                        if(values.size != int(numGroups)):
                                raise ValueError(f"{name} must be a scalar or contain exactly {numGroups} entries.")

                if(np.any(values < 1)):
                        raise ValueError(f"{name} must contain only values >= 1.")

                return values

        @staticmethod
        def torchNormalizeIntensityContributions(intensity_stack, eps = 1e-12):
                values = torch.clamp(intensity_stack, min=0.0)
                total = torch.clamp(torch.sum(values, dim=0, keepdim=True), min=float(eps))
                return values / total

        @staticmethod
        def normalizeIntensityContributions(intensity_stack, eps = 1e-12):
                values = np.maximum(np.asarray(intensity_stack, dtype=np.float64), 0.0)
                total = np.maximum(np.sum(values, axis=0, keepdims=True), float(eps))
                return values / total

        @staticmethod
        def computeMSE(image, target):
                return float(np.mean((image - target)**2))

        @staticmethod
        def computePSNR(mse, peakValue = 1.0, eps = 1e-12):
                mse = float(np.asarray(mse, dtype=np.float64).reshape(-1)[0])
                return float(20*np.log10(peakValue / np.sqrt(mse + eps)))

        @staticmethod
        def normalizeWavelengthArray(wavelengths, name = "wavelengths"):
                if(wavelengths is None):
                        return None

                values = np.asarray(wavelengths, dtype=np.float64).reshape(-1)
                if(values.size == 0):
                        raise ValueError(f"{name} must contain at least one wavelength.")
                if(np.any(values <= 0)):
                        raise ValueError(f"{name} must contain only positive wavelengths.")

                if(np.max(values) > 1e-6):
                        values = values * 1e-9

                return values

        @staticmethod
        def buildVisibleTemporalFrameDevices(DeviceDictionary, frameLUTWavelengths, frameIlluminationWavelengths = None):
                frame_lut_m = CGHGenerator.normalizeWavelengthArray(frameLUTWavelengths, "frameLUTWavelengths")
                if(frameIlluminationWavelengths is None):
                        frame_illum_m = frame_lut_m.copy()
                else:
                        frame_illum_m = CGHGenerator.normalizeWavelengthArray(
                                frameIlluminationWavelengths,
                                "frameIlluminationWavelengths"
                        )
                        if(frame_illum_m.size != frame_lut_m.size):
                                raise ValueError("frameIlluminationWavelengths must match frameLUTWavelengths in length.")

                if(DeviceDictionary["device"] not in ("0.47", "0.67", "0.98")):
                        raise ValueError(
                                "Visible temporal wavelength multiplexing currently supports the visible empirical "
                                "devices 0.47, 0.67, and 0.98."
                        )

                frame_devices = []
                frame_pLevels = []
                for lut_m, illum_m in zip(frame_lut_m, frame_illum_m):
                        frame_device = DeviceLibrary().defineDevice(
                                DeviceDictionary["device"],
                                discScheme=DeviceDictionary.get("discScheme", "Empirical"),
                                lutWavelength=lut_m
                        )
                        pLevel = np.asarray(frame_device["pLevel"], dtype=np.float64)
                        if(pLevel.ndim != 1):
                                raise ValueError(
                                        "Visible temporal wavelength multiplexing expects a 1D phase LUT per frame."
                                )

                        scale = lut_m / illum_m
                        effective_pLevel = np.remainder(pLevel * scale, 1.0)
                        scaled_endpoint = float(pLevel[-1] * scale)
                        if(np.isclose(scaled_endpoint, 1.0, atol=1e-9)):
                                effective_pLevel[-1] = 1.0
                        else:
                                effective_pLevel[-1] = np.remainder(scaled_endpoint, 1.0)
                        frame_devices.append(frame_device)
                        frame_pLevels.append(effective_pLevel)

                return {
                        "frame_devices": frame_devices,
                        "frame_pLevels": frame_pLevels,
                        "frame_lut_m": frame_lut_m,
                        "frame_illum_m": frame_illum_m
                }

        # Reads an image file and stores to imTarget        
        def loadTarget(self, filename = "", colorChannel = 0):
                # Either load the image through calling or through TK
                if filename != "":
                        fn = filename
                else:
                        Tk().withdraw()
                        fn = askopenfilename(title="Select an image file", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")])
                # Read in image file
                I = cv2.imread(fn)
                if(I is None):
                        raise FileNotFoundError(f"Could not read image file: {fn}")

                # Convert image to float (double in Matlab) and normalize to [0,1]
                I = I.astype(np.float64) / 255.0
                # Select color channel (0,1,2)/(R,G,B)
                I = I[:,:, colorChannel]
                # Assign properties
                self.imTarget_orig = I

                self.imTarget = self.imTarget_orig      

        # Resize target image to [self.imTarget.shape[0],self.imTarget.shape[1]]
        def resizeTarget(self, lambda_m, h, w, pitchW, pitchH, ShiftFOV = True, preserveAspect = True):
                I0 = self.imTarget
                S0 = np.shape(I0)

                if(preserveAspect):
                        FOV_aspect_ratio = np.arcsin(lambda_m/pitchW)/np.arcsin(lambda_m/pitchH)

                        # Zero pads image to match the target aspect ratio. Needed due to the FOV being dictated by pixel dimensions
                        # Need to pad veritcally to meet aspect ratio
                        if(S0[1]/S0[0] < FOV_aspect_ratio):
                                S0_delta = int(abs(S0[0] - np.round(S0[1]*FOV_aspect_ratio))) # Must specifcy int to avoid error with pad

                                if(np.mod(S0_delta,2) == 0):
                                        # Pad S0_delta//2 on top and bottom of I0
                                        I1 = np.pad(I0, ((0,0), (S0_delta//2, S0_delta//2)), 'constant')
                                else:
                                        # Pad S0_delta//2 + 1/2 on top and S0_delta//2 - 1/2 on bottom of I0
                                        I1 = np.pad(I0, ((0,0), (int(float(S0_delta)//2 + 1/2) + 1, int(float(S0_delta)//2 - 1/2) + 1)), 'constant')
                        # Need to pad horizantally to meet aspect ratio
                        elif(S0[1]/S0[0] > FOV_aspect_ratio):
                                S0_delta = int(abs(S0[0] - np.round(S0[1]/FOV_aspect_ratio))) # Must specifcy int to avoid error with pad

                                if(np.mod(S0_delta,2) == 0):
                                        # Pad S0_delta//2 on right and left of I0
                                        I1 = np.pad(I0, ((S0_delta//2, S0_delta//2), (0,0)), 'constant')
                                else:
                                        # Pad S0_delta//2 + 1/2 on top and S0_delta//2 - 1/2 on bottom of I0
                                        I1 = np.pad(I0, (((int(float(S0_delta)//2 + 1/2) + 1, int(float(S0_delta)//2 - 1/2) + 1)), (0,0)), 'constant')
                        else:
                                I1 = I0
                else:
                        I1 = I0
                
                # Shift image to be centered on 0th order if shiftFOV is true
                if ShiftFOV:
                        # Must cast to int for roll to avoid error
                        horz_shift = int(np.round((np.shape(I1)[0])/2))
                        vert_shift = int(np.round((np.shape(I1)[1])/2))
                        I2 = np.roll(I1, shift=(horz_shift, vert_shift), axis=(0,1))
                else:
                        I2 = I1

                # Resize the square image to match PLM array size for GS iteration.
                # Image will be distorted for the GS iteration but the ouptut from 
                # the PLM will fit the square FOV and maintain original aspect ratio.
                
                # Reshape to the correct dimensions
                I3 = cv2.resize(I2, (w, h), interpolation=cv2.INTER_NEAREST)
                # Normalize to [0,1]
                I3_min = np.min(I3)
                I3_max = np.max(I3)
                if(np.isclose(I3_max, I3_min)):
                        I3 = np.zeros_like(I3, dtype=np.float64)
                else:
                        I3 = (I3 - I3_min)/(I3_max - I3_min)
                # Set the target image
                self.imTarget = I3

        # Update modifications to target image (invert, flip, etc.)
        def updateTarget(self, FlipUD = False, FlipLR = False, invertTarget = False):
                if(FlipUD):
                        I = np.flipud(self.imTarget)
                        self.imTarget = I.copy()
                if(FlipLR):
                        I = np.fliplr(self.imTarget)
                        self.imTarget = I.copy()
                if(invertTarget):
                        self.imTarget = 1 - self.imTarget.copy()
                
        # Create CGH for target image
        def createCGH(self, DeviceDictionary, 
                      filename = "", bitPlanes = 1, colorChannel = 0,
                      FlipUD = False, FlipLR = False, InvertTarget = False,
                      alg = 'GS', numIter = 1, initialPhase = 'Random', propMethod = 'Fourier',
                      ShiftFOV = True, showImages = False, pixelGrouping = 1, algParams = None,
                      preserveAspect = True):

                # Show images setting enabled
                self.show_images = showImages
                
                # Set number of bitplanes in CGH
                self.bitPlanes = bitPlanes

                if(algParams is None):
                        algParams = {}
                skipRecovery = bool(algParams.get("skipRecovery", False))
                fastTemporalMapping = bool(algParams.get("fastTemporalMapping", False))
                deferTemporalMapping = bool(algParams.get("deferTemporalMapping", False))
                frameLUTWavelengths = algParams.get("frameLUTWavelengths", None)
                if(frameLUTWavelengths is not None):
                        frameLUTWavelengths = np.asarray(frameLUTWavelengths).reshape(-1)
                        temporalFrames = int(max(1, algParams.get("temporalFrames", len(frameLUTWavelengths))))
                        if(len(frameLUTWavelengths) != temporalFrames):
                                raise ValueError("temporalFrames must match the number of frameLUTWavelengths entries.")
                else:
                        temporalFrames = int(max(1, algParams.get("temporalFrames", 1)))
                if(temporalFrames > 1 and self.bitPlanes != 1):
                        raise ValueError("Temporal multiplexing currently expects bitPlanes=1.")
                if(temporalFrames > 1 and alg.upper() not in ("ADAM", "ADAMTEMPORAL", "ADAMWGS", "ADAMWGSTEMPORAL", "ADAMWGSVISIBLETEMPORAL")):
                        raise ValueError(
                                "Joint temporal multiplexing in TIPLMSuiteMulti is currently implemented for "
                                "ADAM, ADAMWGS, and ADAMWGSVISIBLETEMPORAL."
                        )

                # Reset outputs for repeated calls with the same object
                self.CGH_mapped = None
                self.CGH_phase = None
                self.imRecovered_cont = None
                self.imRecovered_disc = None
                self.optimizedIllumination = None
                self.illuminationWasOptimized = False
                self.CGH_output_cont = None
                self.CGH_output_cont_subframes = None
                self.CGH_output_disc = None
                self.CGH_output_phase_disc = None
                self.CGH_output_state_disc = None
                self.CGH_output_phase_disc_subframes = None
                self.CGH_output_state_disc_subframes = None
                self.CGH_effective_field_cont = None
                self.CGH_effective_field_disc = None
                self.CGH_effective_field_cont_subframes = None
                self.CGH_effective_field_disc_subframes = None
                self.temporalFrameLUTWavelengths = None
                self.temporalFrameIlluminationWavelengths = None
                self.CGH_is_sequence = False
                self.imRecovered_cont_native = None
                self.imRecovered_disc_native = None
                self.softWTAGroupLUTWavelengths = None
                self.softWTAGroupTemporalFrames = None
                self.softWTASelectorWeights = None
                self.softWTAHardWinnerMap = None
                self.softWTASoftFusedNative = None
                self.softWTAHardFusedNative = None
                self.softWTAChosenFusionMode = None
                self.softWTAChosenFusionMSE = None
                self.softWTAChosenFusionPSNR = None
                self.softWTAGroupReconsNative = None
                self.softWTAGroupOutputs = None
                self.outputShape = (DeviceDictionary["h"], DeviceDictionary["w"])

                group_h, group_w = CGHGenerator.normalizePixelGrouping(pixelGrouping)
                if(DeviceDictionary["h"] % group_h != 0 or DeviceDictionary["w"] % group_w != 0):
                        raise ValueError(
                                "pixelGrouping must evenly divide the device size. "
                                f"Got pixelGrouping={(group_h, group_w)} for device size "
                                f"{(DeviceDictionary['h'], DeviceDictionary['w'])}."
                        )

                # Load target image to compute CGH
                self.loadTarget(filename, colorChannel)
                self.pitchW = DeviceDictionary["pitchW"]
                self.pitchH = DeviceDictionary["pitchH"]

                # Resize the target image using the effective grouped-pixel grid.
                eff_h = DeviceDictionary["h"] // group_h
                eff_w = DeviceDictionary["w"] // group_w
                eff_pitchH = DeviceDictionary["pitchH"] * group_h
                eff_pitchW = DeviceDictionary["pitchW"] * group_w

                self.resizeTarget(self.lambda_m, eff_h, eff_w, eff_pitchW, eff_pitchH, ShiftFOV, preserveAspect=preserveAspect)

                # Update the target image with any modifications (FlipLR, Flip UD, Invert)
                self.updateTarget(FlipUD, FlipLR, InvertTarget)
                self.imTarget_recon = self.imTarget.copy()

                for i in range(0,self.bitPlanes):
                        self.CGH_output_cont_subframes = None
                        self.CGH_output_phase_disc_subframes = None
                        self.CGH_output_state_disc_subframes = None
                        self.CGH_effective_field_cont_subframes = None
                        self.CGH_effective_field_disc_subframes = None
                        self.RunPhaseAlg(alg, initialPhase, numIter, propMethod, DeviceDictionary, algParams, (group_h, group_w))

                        if(deferTemporalMapping and self.CGH_output_cont_subframes is not None):
                                phase_stack = np.asarray(self.CGH_output_cont_subframes)
                                if(phase_stack.ndim == 3 and phase_stack.shape[0] > 1):
                                        self.CGH_is_sequence = True
                                        self.CGH_output_cont_subframes = np.mod(phase_stack, 2*np.pi).astype(np.float32, copy=False)
                                        self.CGH_output_cont = self.CGH_output_cont_subframes[0].copy()
                                        self.CGH_mapped = None
                                        self.CGH_phase = None
                                        continue

                        if(
                                self.CGH_output_phase_disc_subframes is not None
                                and self.CGH_output_state_disc_subframes is not None
                        ):
                                phase_stack = np.asarray(self.CGH_output_phase_disc_subframes)
                                state_stack = np.asarray(self.CGH_output_state_disc_subframes)
                                cont_stack = (
                                        np.asarray(self.CGH_output_cont_subframes)
                                        if self.CGH_output_cont_subframes is not None
                                        else phase_stack.copy()
                                )
                                eff_cont_stack = (
                                        np.asarray(self.CGH_effective_field_cont_subframes)
                                        if self.CGH_effective_field_cont_subframes is not None
                                        else None
                                )
                                eff_disc_stack = (
                                        np.asarray(self.CGH_effective_field_disc_subframes)
                                        if self.CGH_effective_field_disc_subframes is not None
                                        else eff_cont_stack
                                )

                                if(phase_stack.ndim != 3 or state_stack.ndim != 3 or phase_stack.shape != state_stack.shape):
                                        raise ValueError(
                                                "Temporal output must provide matching [frames, height, width] "
                                                "phase/state stacks."
                                        )

                                if(phase_stack.shape[0] > 1):
                                        self.CGH_is_sequence = True

                                phase_cont_subframes = []
                                phase_disc_subframes = []
                                state_disc_subframes = []
                                effective_cont_subframes = []
                                effective_disc_subframes = []
                                fast_stack_mapping = fastTemporalMapping and phase_stack.shape[0] > 1

                                for frame_idx in range(phase_stack.shape[0]):
                                        self.CGH_output_cont = cont_stack[frame_idx].copy()
                                        self.CGH_output_disc = phase_stack[frame_idx].copy()
                                        self.CGH_output_phase_disc = phase_stack[frame_idx].copy()
                                        self.CGH_output_state_disc = state_stack[frame_idx].copy()
                                        self.CGH_effective_field_cont = (
                                                eff_cont_stack[frame_idx].copy() if eff_cont_stack is not None else None
                                        )
                                        self.CGH_effective_field_disc = (
                                                eff_disc_stack[frame_idx].copy()
                                                if eff_disc_stack is not None
                                                else self.CGH_effective_field_cont
                                        )

                                        if(group_h != 1 or group_w != 1):
                                                self.CGH_output_cont = CGHGenerator.expandGroupedImage(self.CGH_output_cont, (group_h, group_w))
                                                self.CGH_output_phase_disc = CGHGenerator.expandGroupedImage(self.CGH_output_phase_disc, (group_h, group_w))
                                                self.CGH_output_state_disc = CGHGenerator.expandGroupedImage(self.CGH_output_state_disc, (group_h, group_w))
                                                self.imTarget_recon = CGHGenerator.expandGroupedImage(self.imTarget, (group_h, group_w))
                                                self.CGH_output_disc = self.CGH_output_phase_disc.copy()

                                        if(
                                                not skipRecovery
                                                and (
                                                self.CGH_effective_field_cont is None
                                                or self.CGH_effective_field_disc is None
                                                or self.CGH_effective_field_cont.shape != self.CGH_output_cont.shape
                                                or self.CGH_effective_field_disc.shape != self.CGH_output_phase_disc.shape
                                                )
                                        ):
                                                self._updateEffectiveFieldsForCurrentOutput()

                                        if(not skipRecovery):
                                                self.recoverImg(ShiftFOV)

                                        if(not fast_stack_mapping):
                                                CGH_plane_mapped = self.deviceLibary.formatPLM(DeviceDictionary, self.CGH_output_state_disc)

                                                if(self.CGH_mapped is None):
                                                        self.CGH_mapped = np.zeros((CGH_plane_mapped.shape[0], CGH_plane_mapped.shape[1], 0))
                                                if(self.CGH_phase is None):
                                                        self.CGH_phase = np.zeros((self.CGH_output_phase_disc.shape[0], self.CGH_output_phase_disc.shape[1], 0))

                                                self.CGH_mapped = np.concatenate((self.CGH_mapped, CGH_plane_mapped[:, :, np.newaxis]), axis=2)
                                                self.CGH_phase = np.concatenate((self.CGH_phase, self.CGH_output_phase_disc[:, :, np.newaxis]), axis=2)

                                        if(phase_stack.shape[0] > 1):
                                                phase_cont_subframes.append(self.CGH_output_cont.copy())
                                                phase_disc_subframes.append(self.CGH_output_phase_disc.copy())
                                                state_disc_subframes.append(self.CGH_output_state_disc.copy())
                                                if(self.CGH_effective_field_cont is not None and self.CGH_effective_field_disc is not None):
                                                        effective_cont_subframes.append(self.CGH_effective_field_cont.copy())
                                                        effective_disc_subframes.append(self.CGH_effective_field_disc.copy())

                                if(phase_stack.shape[0] > 1):
                                        self.CGH_output_cont_subframes = np.asarray(phase_cont_subframes, dtype=np.float64)
                                        self.CGH_output_phase_disc_subframes = np.asarray(phase_disc_subframes, dtype=np.float64)
                                        self.CGH_output_state_disc_subframes = np.asarray(state_disc_subframes, dtype=np.uint8)
                                        if(len(effective_cont_subframes) == phase_stack.shape[0]):
                                                self.CGH_effective_field_cont_subframes = np.asarray(effective_cont_subframes, dtype=np.complex128)
                                                self.CGH_effective_field_disc_subframes = np.asarray(effective_disc_subframes, dtype=np.complex128)
                                        self.CGH_output_cont = self.CGH_output_cont_subframes[0].copy()
                                        self.CGH_output_disc = self.CGH_output_phase_disc_subframes[0].copy()
                                        self.CGH_output_phase_disc = self.CGH_output_phase_disc_subframes[0].copy()
                                        self.CGH_output_state_disc = self.CGH_output_state_disc_subframes[0].copy()
                                        if(fast_stack_mapping):
                                                self.CGH_mapped = self.deviceLibary.formatPLMStack(
                                                        DeviceDictionary, self.CGH_output_state_disc_subframes
                                                )
                                                self.CGH_phase = np.ascontiguousarray(
                                                        np.moveaxis(self.CGH_output_phase_disc_subframes, 0, 2)
                                                )
                                continue

                        phase_planes = []
                        phase_cont_subframes = []
                        phase_disc_subframes = []
                        state_disc_subframes = []
                        effective_cont_subframes = []
                        effective_disc_subframes = []

                        if(self.CGH_output_cont_subframes is not None):
                                phase_stack = np.asarray(self.CGH_output_cont_subframes)
                                if(phase_stack.ndim == 3):
                                        for frame_idx in range(phase_stack.shape[0]):
                                                phase_planes.append(np.mod(phase_stack[frame_idx], 2*np.pi))
                                        if(phase_stack.shape[0] > 1):
                                                self.CGH_is_sequence = True

                        if(len(phase_planes) == 0):
                                phase_planes.append(np.mod(self.CGH_output_cont, 2*np.pi))

                        fast_stack_mapping = fastTemporalMapping and len(phase_planes) > 1

                        for phase_plane in phase_planes:
                                self.CGH_output_cont = phase_plane

                                # Discretize phase on the effective grid
                                [self.CGH_output_phase_disc, self.CGH_output_state_disc] = self.discretePhase(
                                        self.CGH_output_cont, DeviceDictionary["nLevel"], DeviceDictionary["pLevel"]
                                )
                                self.CGH_output_phase_disc = 2*np.pi*self.CGH_output_phase_disc

                                # Expand the grouped-pixel hologram back to the physical PLM grid
                                if(group_h != 1 or group_w != 1):
                                        self.CGH_output_cont = CGHGenerator.expandGroupedImage(self.CGH_output_cont, (group_h, group_w))
                                        self.CGH_output_phase_disc = CGHGenerator.expandGroupedImage(self.CGH_output_phase_disc, (group_h, group_w))
                                        self.CGH_output_state_disc = CGHGenerator.expandGroupedImage(self.CGH_output_state_disc, (group_h, group_w))
                                        self.imTarget_recon = CGHGenerator.expandGroupedImage(self.imTarget, (group_h, group_w))

                                self.CGH_output_disc = self.CGH_output_phase_disc.copy()
                                if(not skipRecovery):
                                        self._updateEffectiveFieldsForCurrentOutput()

                                        # Recover image for debug/visualization purposes
                                        self.recoverImg(ShiftFOV)

                                if(not fast_stack_mapping):
                                        CGH_plane_mapped = self.deviceLibary.formatPLM(DeviceDictionary, self.CGH_output_state_disc)

                                        if(self.CGH_mapped is None):
                                                self.CGH_mapped = np.zeros((CGH_plane_mapped.shape[0], CGH_plane_mapped.shape[1], 0))
                                        if(self.CGH_phase is None):
                                                self.CGH_phase = np.zeros((self.CGH_output_phase_disc.shape[0],self.CGH_output_phase_disc.shape[1], 0)) 

                                        self.CGH_mapped = np.concatenate((self.CGH_mapped, CGH_plane_mapped[:, :, np.newaxis]), axis=2)
                                        self.CGH_phase = np.concatenate((self.CGH_phase, self.CGH_output_phase_disc[:, :, np.newaxis]), axis=2)

                                if(len(phase_planes) > 1):
                                        phase_cont_subframes.append(self.CGH_output_cont.copy())
                                        phase_disc_subframes.append(self.CGH_output_phase_disc.copy())
                                        state_disc_subframes.append(self.CGH_output_state_disc.copy())
                                        if(self.CGH_effective_field_cont is not None and self.CGH_effective_field_disc is not None):
                                                effective_cont_subframes.append(self.CGH_effective_field_cont.copy())
                                                effective_disc_subframes.append(self.CGH_effective_field_disc.copy())

                        if(len(phase_cont_subframes) > 1):
                                self.CGH_output_cont_subframes = np.asarray(phase_cont_subframes, dtype=np.float64)
                                self.CGH_output_phase_disc_subframes = np.asarray(phase_disc_subframes, dtype=np.float64)
                                self.CGH_output_state_disc_subframes = np.asarray(state_disc_subframes, dtype=np.uint8)
                                if(len(effective_cont_subframes) == len(phase_cont_subframes)):
                                        self.CGH_effective_field_cont_subframes = np.asarray(effective_cont_subframes, dtype=np.complex128)
                                        self.CGH_effective_field_disc_subframes = np.asarray(effective_disc_subframes, dtype=np.complex128)
                                self.CGH_output_cont = self.CGH_output_cont_subframes[0].copy()
                                self.CGH_output_disc = self.CGH_output_phase_disc_subframes[0].copy()
                                self.CGH_output_phase_disc = self.CGH_output_phase_disc_subframes[0].copy()
                                self.CGH_output_state_disc = self.CGH_output_state_disc_subframes[0].copy()
                                if(fast_stack_mapping):
                                        self.CGH_mapped = self.deviceLibary.formatPLMStack(
                                                DeviceDictionary, self.CGH_output_state_disc_subframes
                                        )
                                        self.CGH_phase = np.ascontiguousarray(
                                                np.moveaxis(self.CGH_output_phase_disc_subframes, 0, 2)
                                        )
                
                if(self.bitPlanes != 1 and not self.CGH_is_sequence):
                        self.stackPlanes()
                elif(self.CGH_is_sequence and not skipRecovery):
                        self.recoverTemporalSequence(ShiftFOV)
        
        # Format discretized final phase hologram for bitPlanes > 3
        def stackPlanes(self):
                bitPlanes = self.bitPlanes
                bitDepth = int(bitPlanes/3)
                CGH_mapped_RGB = np.zeros((self.CGH_mapped.shape[0], self.CGH_mapped.shape[1], 3))
                for colors in range(3): # Loop over RGB channels (0,1,2) for 3 color channels
                        CGH_mapped_color = np.zeros((self.CGH_mapped.shape[0], self.CGH_mapped.shape[1]))
                        for bit in range(bitDepth):
                                CGH_mapped_color = CGH_mapped_color + (self.CGH_mapped[:,:,colors*bitDepth + bit] != 0)*(2**(bit))
                        CGH_mapped_RGB[:,:,colors] = CGH_mapped_color
                
                if(bitPlanes == 48):
                        CGH_output = np.uint16(CGH_mapped_RGB)
                elif(bitPlanes == 3):
                        CGH_output = CGH_mapped_RGB
                else:
                        CGH_output = np.uint8(CGH_mapped_RGB)

                self.CGH_mapped = CGH_output

        # Recover PLM Display Image. Debug purposes only
        def recoverImg(self, ShiftFOV = True):
                if(self.imTarget is not None):
                        if(self.CGH_effective_field_cont is not None):
                                E_ip_cont = self.prop("forward", self.CGH_effective_field_cont, "FOURIER")
                                Irecon_alg_cont = CGHGenerator.fieldToIntensity(E_ip_cont)
                                target_ref_cont = self.getReconstructionTarget(Irecon_alg_cont.shape)
                                _, Irecon_alg_cont = CGHGenerator.scaleIntensityToTarget(Irecon_alg_cont, target_ref_cont)
                                Erecon_alg_cont = E_ip_cont
                        else:
                                Irecon_alg_cont, Erecon_alg_cont = self.reconstruct(self.CGH_output_cont)

                        if(self.CGH_effective_field_disc is not None):
                                E_ip_disc = self.prop("forward", self.CGH_effective_field_disc, "FOURIER")
                                Irecon_alg_disc = CGHGenerator.fieldToIntensity(E_ip_disc)
                                target_ref_disc = self.getReconstructionTarget(Irecon_alg_disc.shape)
                                _, Irecon_alg_disc = CGHGenerator.scaleIntensityToTarget(Irecon_alg_disc, target_ref_disc)
                                Erecon_alg_disc = E_ip_disc
                        else:
                                Irecon_alg_disc, Erecon_alg_disc = self.reconstruct(self.CGH_output_phase_disc)

                        if (ShiftFOV):
                                # Must cast to int for roll to avoid error
                                shift_cont = (int(Irecon_alg_cont.shape[0]/2), int(Irecon_alg_cont.shape[1]/2))
                                shift_disc = (int(Irecon_alg_disc.shape[0]/2), int(Irecon_alg_disc.shape[1]/2))
                                Irecon_alg_cont = np.roll(Irecon_alg_cont, shift=shift_cont, axis=(0,1))
                                Erecon_alg_cont = np.roll(Erecon_alg_cont, shift=shift_cont, axis=(0,1))
                                Irecon_alg_disc = np.roll(Irecon_alg_disc, shift=shift_disc, axis=(0,1))
                                Erecon_alg_disc = np.roll(Erecon_alg_disc, shift=shift_disc, axis=(0,1))
                        
                        # Store the native reconstruction resolution for analysis.
                        self.imRecovered_cont_native = Irecon_alg_cont.copy()
                        self.imRecovered_disc_native = Irecon_alg_disc.copy()

                        # For viewing, preserve the physical PLM output size even when the
                        # effective information content is reduced by grouped-pixel encoding.
                        display_shape = getattr(self, "outputShape", Irecon_alg_cont.shape)
                        if(Irecon_alg_cont.shape != display_shape):
                                self.imRecovered_cont = cv2.resize(
                                        Irecon_alg_cont,
                                        (display_shape[1], display_shape[0]),
                                        interpolation=cv2.INTER_NEAREST
                                )
                        else:
                                self.imRecovered_cont = Irecon_alg_cont

                        if(Irecon_alg_disc.shape != display_shape):
                                self.imRecovered_disc = cv2.resize(
                                        Irecon_alg_disc,
                                        (display_shape[1], display_shape[0]),
                                        interpolation=cv2.INTER_NEAREST
                                )
                        else:
                                self.imRecovered_disc = Irecon_alg_disc

        def recoverTemporalSequence(self, ShiftFOV = True):
                cont_stack = self.CGH_effective_field_cont_subframes
                disc_stack = self.CGH_effective_field_disc_subframes

                if(cont_stack is not None):
                        cont_stack = np.asarray(cont_stack)
                if(disc_stack is not None):
                        disc_stack = np.asarray(disc_stack)

                if(cont_stack is None and disc_stack is None):
                        if(self.CGH_phase is None or len(self.CGH_phase.shape) != 3 or self.CGH_phase.shape[2] == 0):
                                return
                        phase_stack = np.moveaxis(self.CGH_phase, 2, 0)
                else:
                        phase_stack = None

                if(cont_stack is None and disc_stack is not None):
                        cont_stack = disc_stack
                if(disc_stack is None and cont_stack is not None):
                        disc_stack = cont_stack

                if(cont_stack is not None):
                        n_frames = cont_stack.shape[0]
                elif(disc_stack is not None):
                        n_frames = disc_stack.shape[0]
                else:
                        n_frames = phase_stack.shape[0]

                recon_power_cont = None
                recon_power_disc = None

                for frame_idx in range(n_frames):
                        if(cont_stack is not None):
                                E_ip_cont = self.prop("forward", cont_stack[frame_idx], "FOURIER")
                                frame_power_cont = CGHGenerator.fieldToIntensity(E_ip_cont)
                        else:
                                E_hp_cont = np.exp(1j*phase_stack[frame_idx])
                                E_ip_cont = self.prop("forward", E_hp_cont, "FOURIER")
                                frame_power_cont = CGHGenerator.fieldToIntensity(E_ip_cont)

                        if(disc_stack is not None):
                                E_ip_disc = self.prop("forward", disc_stack[frame_idx], "FOURIER")
                                frame_power_disc = CGHGenerator.fieldToIntensity(E_ip_disc)
                        else:
                                E_hp_disc = np.exp(1j*phase_stack[frame_idx])
                                E_ip_disc = self.prop("forward", E_hp_disc, "FOURIER")
                                frame_power_disc = CGHGenerator.fieldToIntensity(E_ip_disc)

                        if(recon_power_cont is None):
                                recon_power_cont = np.zeros_like(frame_power_cont, dtype=np.float64)
                        if(recon_power_disc is None):
                                recon_power_disc = np.zeros_like(frame_power_disc, dtype=np.float64)

                        recon_power_cont = recon_power_cont + frame_power_cont
                        recon_power_disc = recon_power_disc + frame_power_disc

                recon_power_cont = recon_power_cont / float(max(n_frames, 1))
                recon_power_disc = recon_power_disc / float(max(n_frames, 1))

                Irecon_alg_cont = np.maximum(recon_power_cont, 0)
                Irecon_alg_disc = np.maximum(recon_power_disc, 0)

                target_ref = self.getReconstructionTarget(Irecon_alg_cont.shape)
                _, Irecon_alg_cont = CGHGenerator.scaleIntensityToTarget(Irecon_alg_cont, target_ref)
                _, Irecon_alg_disc = CGHGenerator.scaleIntensityToTarget(Irecon_alg_disc, target_ref)

                if (ShiftFOV):
                        shift_cont = (int(Irecon_alg_cont.shape[0]/2), int(Irecon_alg_cont.shape[1]/2))
                        shift_disc = (int(Irecon_alg_disc.shape[0]/2), int(Irecon_alg_disc.shape[1]/2))
                        Irecon_alg_cont = np.roll(Irecon_alg_cont, shift=shift_cont, axis=(0,1))
                        Irecon_alg_disc = np.roll(Irecon_alg_disc, shift=shift_disc, axis=(0,1))

                self.imRecovered_cont_native = Irecon_alg_cont.copy()
                self.imRecovered_disc_native = Irecon_alg_disc.copy()

                display_shape = getattr(self, "outputShape", Irecon_alg_cont.shape)
                if(Irecon_alg_cont.shape != display_shape):
                        self.imRecovered_cont = cv2.resize(
                                Irecon_alg_cont,
                                (display_shape[1], display_shape[0]),
                                interpolation=cv2.INTER_NEAREST
                        )
                else:
                        self.imRecovered_cont = Irecon_alg_cont

                if(Irecon_alg_disc.shape != display_shape):
                        self.imRecovered_disc = cv2.resize(
                                Irecon_alg_disc,
                                (display_shape[1], display_shape[0]),
                                interpolation=cv2.INTER_NEAREST
                        )
                else:
                        self.imRecovered_disc = Irecon_alg_disc

################################
##       CGH functions        ##
##  GS/GradientDescent Algs   ##
################################

        def RunPhaseAlg(self, alg = 'GS', initialPhase = 'Random', numIter = 1, propMethod = 'Fourier', DeviceDictionary = None, algParams = None, pixelGrouping = (1, 1)):

                if(algParams is None):
                        algParams = {}

                # Account for illumination
                self.configureIlluminationFromParams(algParams)

                if(alg is None):
                        self.CGH_output = np.zeros((self.imTarget.shape[0],self.imTarget.shape[1]))
                        self.alg_phase_disc = np.zeros((self.imTarget.shape[0],self.imTarget.shape[1]))
                        self.ReconField_cont = np.zeros((self.imTarget.shape[0],self.imTarget.shape[1]))
                        self.ReconIntensity_cont = np.zeros((self.imTarget.shape[0],self.imTarget.shape[1]))
                        self.ReconField_disc = np.zeros((self.imTarget.shape[0],self.imTarget.shape[1]))
                        self.ReconIntensity_disc = np.zeros((self.imTarget.shape[0],self.imTarget.shape[1]))
                        self.ReconIntensity_cont_FOV = np.zeros(np.shape(self.imTarget_orig_FOV))
                        self.ReconIntensity_disc_FOV = np.zeros(np.shape(self.imTarget_orig_FOV))
                elif(alg.upper() == "GS"):
                        self.runGS(initialPhase, numIter, propMethod)
                elif(alg.upper() == "WCGS"):
                        self.runWCGS(initialPhase, numIter, propMethod)
                elif(alg.upper() == "AWCGS"):
                        self.runAWCGS(initialPhase, numIter, propMethod)
                elif(alg.upper() == "ADAM"):
                        temporalFrames = int(max(1, algParams.get("temporalFrames", 1)))
                        if(temporalFrames > 1):
                                self.runADAMTemporal(
                                        nLevel=DeviceDictionary["nLevel"],
                                        pLevel=DeviceDictionary["pLevel"],
                                        temporalFrames=temporalFrames,
                                        initialPhase=initialPhase,
                                        numIter=numIter,
                                        propMethod=propMethod,
                                        temporalSeedJitter=algParams.get("temporalSeedJitter", 0.10),
                                        algParams=algParams
                                )
                        else:
                                self.runADAM(initialPhase, numIter, propMethod, algParams=algParams)
                elif(alg.upper() == "ADAMTEMPORAL"):
                        self.runADAMTemporal(
                                nLevel=DeviceDictionary["nLevel"],
                                pLevel=DeviceDictionary["pLevel"],
                                temporalFrames=int(max(1, algParams.get("temporalFrames", 1))),
                                initialPhase=initialPhase,
                                numIter=numIter,
                                propMethod=propMethod,
                                temporalSeedJitter=algParams.get("temporalSeedJitter", 0.10),
                                algParams=algParams
                        )
                elif(alg.upper() == "ADAMWGS"):
                        temporalFrames = int(max(1, algParams.get("temporalFrames", 1)))
                        if(algParams.get("frameLUTWavelengths", None) is not None):
                                self.runADAMWGSVisibleTemporal(
                                        DeviceDictionary,
                                        frameLUTWavelengths=algParams.get("frameLUTWavelengths", None),
                                        frameIlluminationWavelengths=algParams.get("frameIlluminationWavelengths", None),
                                        initialPhase=initialPhase,
                                        numIter=numIter,
                                        propMethod=propMethod,
                                        temporalSeedJitter=algParams.get("temporalSeedJitter", 0.10),
                                        algParams=algParams
                                )
                        elif(temporalFrames > 1):
                                self.runADAMWGSTemporal(
                                        DeviceDictionary["nLevel"],
                                        DeviceDictionary["pLevel"],
                                        temporalFrames=temporalFrames,
                                        initialPhase=initialPhase,
                                        numIter=numIter,
                                        propMethod=propMethod,
                                        temporalSeedJitter=algParams.get("temporalSeedJitter", 0.10),
                                        algParams=algParams
                                )
                        else:
                                self.runADAMwGS(DeviceDictionary["nLevel"], DeviceDictionary["pLevel"], initialPhase, numIter, propMethod, algParams=algParams)
                elif(alg.upper() == "ADAMWGSTEMPORAL"):
                        self.runADAMWGSTemporal(
                                DeviceDictionary["nLevel"],
                                DeviceDictionary["pLevel"],
                                temporalFrames=int(max(1, algParams.get("temporalFrames", 1))),
                                initialPhase=initialPhase,
                                numIter=numIter,
                                propMethod=propMethod,
                                temporalSeedJitter=algParams.get("temporalSeedJitter", 0.10),
                                algParams=algParams
                        )
                elif(alg.upper() == "ADAMWGSVISIBLETEMPORAL"):
                        self.runADAMWGSVisibleTemporal(
                                DeviceDictionary,
                                frameLUTWavelengths=algParams.get("frameLUTWavelengths", None),
                                frameIlluminationWavelengths=algParams.get("frameIlluminationWavelengths", None),
                                initialPhase=initialPhase,
                                numIter=numIter,
                                propMethod=propMethod,
                                temporalSeedJitter=algParams.get("temporalSeedJitter", 0.10),
                                algParams=algParams
                        )
                elif(alg.upper() == "MULTICOLOR"):
                        self.runMultiColor(
                                P=int(algParams.get("P", 3)),
                                T=int(algParams.get("T", 3)),
                                lambda_p=algParams.get("lambda_p", [532e-9, 532e-9, 532e-9]),
                                lambda_p_anch=algParams.get("lambda_p_anch", 532e-9),
                                initialPhase=initialPhase,
                                numIter=numIter
                        )

        # Runs Gerchberg-Saxton algorithm to generate the object field and create the target image at the image plane
        def runGS(self, initialPhase = 'Random', numIter = 1, propMethod = 'Fourier'):
                phase_ip_init = self.resolveInitialPhaseMap(initialPhase, allowTemporalStack=True)
                
                E0_ip_target = np.sqrt(self.imTarget) # Target field amplitude
                        
                # Calculate initial image with initial phase map
                E_ip_init = E0_ip_target * np.exp(1j * phase_ip_init)
                E_ip_inp = E_ip_init

                # initial MSE
                msel = np.zeros((numIter, 1))

                # peak signal to noise ratio
                psnrl = np.zeros((numIter, 1))

                # Define performance metrics
                best_loss = 1e10
                phase_hp = phase_ip_init
                best_phase = phase_hp
                image_recon = np.zeros(self.imTarget.shape)

                # Peform GS algorithm
                for i in tqdm(range(0, numIter)):
                        # back-propogate from image plane to plm plane (usually FFT)
                        E_hp = self.prop("backward", E_ip_inp, propMethod)
                        phase_hp = np.angle(E_hp)

                        # forward-propogate from plm plane to image plane
                        E_ip = self.prop("forward", self.targetIllumination*np.exp(1j*phase_hp), propMethod)
                        phase_ip = np.angle(E_ip)

                        # Determine the reconstructed image and compare with target
                        image_recon = self.reconstruct(phase_hp, propMethod)[0]

                        # Compare reconstructed image with target image
                        msel[i] = CGHGenerator.computeMSE(image_recon, self.imTarget)
                        psnrl[i] = CGHGenerator.computePSNR(msel[i])

                        # Display the reconstructed image
                        if(self.show_images):
                                cv2.imshow("GS Image" ,image_recon)
                                cv2.setWindowTitle("GS Image", "Iteration: " + str(i) + " MSE: " + str(msel[i]) + " PSNR: " + str(psnrl[i]))
                                cv2.waitKey(0)
                                print(msel[i])
                                print(psnrl[i])

                        # Get the best phase
                        if(msel[i] < best_loss):
                                best_loss = msel[i]
                                best_phase = phase_hp

                        # Apply amplitude constriant at image plane
                        E_ip_inp = E0_ip_target*np.exp(1j*phase_ip)
                
                print("Best MSE: " + str(np.min(msel)))
                print("Best PSNR: " + str(np.max(psnrl)))
                phase_hp = best_phase
                phase_alg_cont = np.mod(phase_hp, 2*np.pi)

                self.CGH_output_cont = phase_alg_cont

        # Run Weighted Constraint GS
        def runWCGS(self, initialPhase = 'Random', numIter = 1, propMethod = 'Fourier', weightedConstraint = 0.1):

                # Additional terms for weighted constraints
                WImage = (cv2.threshold(self.imTarget, weightedConstraint, 255, cv2.THRESH_BINARY)[1]).astype(np.float64)
                WComp = 1 - WImage

                Aholo = np.sqrt(np.sum(self.imTarget**2))/(self.imTarget.shape[1]*self.imTarget.shape[0]) # Amplitude of the hologram
                bk = 10e-8 # convergence factor

                phase_ip_init = self.resolveInitialPhaseMap(initialPhase, allowTemporalStack=True)
                
                E0_ip_target = np.sqrt(self.imTarget) # Target field amplitude
                        
                # Calculate initial image with initial phase map
                E_ip_init = E0_ip_target * np.exp(1j * phase_ip_init)
                E_ip_inp = E_ip_init

                # initial MSE
                msel = np.zeros((numIter, 1))

                # peak signal to noise ratio
                psnrl = np.zeros((numIter, 1))

                # Define performance metrics
                best_loss = 1e10
                phase_hp = phase_ip_init
                best_phase = phase_hp
                image_recon = np.zeros(self.imTarget.shape)

                # Peform WCGS algorithm
                for i in tqdm(range(0, numIter)):
                        # back-propogate from image plane to plm plane (usually FFT)
                        E_hp = self.prop("backward", E_ip_inp, propMethod)
                        phase_hp = np.angle(E_hp)

                        # forward-propogate from plm plane to image plane
                        E_ip = self.prop("forward", Aholo*self.targetIllumination*np.exp(1j*phase_hp), propMethod)
                        E0_ip = np.abs(E_ip)

                        # Determine the reconstructed image and compare with target
                        image_recon = self.reconstruct(phase_hp, propMethod)[0]

                        # Compare reconstructed image with target image
                        msel[i] = CGHGenerator.computeMSE(image_recon, self.imTarget)
                        psnrl[i] = CGHGenerator.computePSNR(msel[i])

                        # Display the reconstructed image
                        if(self.show_images):
                                cv2.imshow("WCGS Image" ,image_recon)
                                cv2.setWindowTitle("WCGS Image", "Iteration: " + str(i+1) + " MSE: " + str(msel[i]) + " PSNR: " + str(psnrl[i]))
                                cv2.waitKey(0)
                                print(msel[i])
                                print(psnrl[i])

                        # Get the best phase
                        if(msel[i] < best_loss):
                                best_loss = msel[i]
                                best_phase = phase_hp

                        # Apply amplitude constriant at image plane
                        ConS = np.abs(E_ip_inp)*(E0_ip_target/E0_ip)**bk # Signal region constraint
                        ConF = E0_ip # Freedom Domain Constraint
                        Con = ConS*WImage + ConF*WComp # Combine constraints
                        E_ip_inp = abs(Con)*E_ip/(E0_ip+10e-10)

                        bk = np.sqrt(bk)
                
                print("Best MSE: " + str(np.min(msel)))
                print("Best PSNR: " + str(np.max(psnrl)))
                phase_hp = best_phase
                phase_alg_cont = np.mod(phase_hp, 2*np.pi)
                self.CGH_output_cont = phase_alg_cont
        
        # Run Adaptive Weighted Constraint GS
        def runAWCGS(self, initialPhase = 'Random', numIter = 1, propMethod = 'Fourier', weightedConstraint = 0.1):

                # Additional terms for weighted constraints
                WImage = (cv2.threshold(self.imTarget, weightedConstraint, 255, cv2.THRESH_BINARY)[1]).astype(np.float64)
                WComp = 1 - WImage

                Aholo = np.sqrt(np.sum(self.imTarget**2))/(self.imTarget.shape[1]*self.imTarget.shape[0]) # Amplitude of the hologram

                phase_ip_init = self.resolveInitialPhaseMap(initialPhase)
                
                E0_ip_target = np.sqrt(self.imTarget) # Target field amplitude
                        
                # Calculate initial image with initial phase map
                E_ip_init = E0_ip_target * np.exp(1j * phase_ip_init)
                E_ip_inp = E_ip_init

                # initial MSE
                msel = np.zeros((numIter, 1))

                # peak signal to noise ratio
                psnrl = np.zeros((numIter, 1))

                # Define performance metrics
                best_loss = 1e10
                phase_hp = phase_ip_init
                best_phase = phase_hp
                image_recon = np.zeros(self.imTarget.shape)

                # Peform AWCGS algorithm
                for i in tqdm(range(0, numIter)):
                        # back-propogate from image plane to plm plane (usually FFT)
                        E_hp = self.prop("backward", E_ip_inp, propMethod)
                        phase_hp = np.angle(E_hp)

                        # forward-propogate from plm plane to image plane
                        E_ip = self.prop("forward", Aholo*self.targetIllumination*np.exp(1j*phase_hp), propMethod)
                        phase_ip = np.angle(E_ip)
                        E0_ip = np.abs(E_ip)

                        # Determine the reconstructed image and compare with target
                        image_recon = self.reconstruct(phase_hp, propMethod)[0]

                        # Compare reconstructed image with target image
                        msel[i] = CGHGenerator.computeMSE(image_recon, self.imTarget)
                        psnrl[i] = CGHGenerator.computePSNR(msel[i])

                        # Display the reconstructed image
                        if(self.show_images):
                                cv2.imshow("AWCGS Image" ,image_recon)
                                cv2.setWindowTitle("AWCGS Image", "Iteration: " + str(i+1) + " MSE: " + str(msel[i]) + " PSNR: " + str(psnrl[i]))
                                cv2.waitKey(0)
                                print(msel[i])
                                print(psnrl[i])

                        # Get the best phase
                        if(msel[i] < best_loss):
                                best_loss = msel[i]
                                best_phase = phase_hp

                        # Apply amplitude constriant at image plane
                        ConS = E0_ip_target*np.exp(E0_ip_target - E0_ip) # Signal region constraint
                        ConF = E0_ip # Freedom Domain Constraint
                        Con = ConS*WImage + ConF*WComp # Combine constraints
                        E_ip_inp = abs(Con)*np.exp(1j*phase_ip)
                
                print("Best MSE: " + str(np.min(msel)))
                print("Best PSNR: " + str(np.max(psnrl)))
                phase_hp = best_phase
                phase_alg_cont = np.mod(phase_hp, 2*np.pi)
                self.CGH_output_cont = phase_alg_cont

        # Run ADAM
        def runADAM(self, initialPhase = 'Random', numIter = 1, propMethod = 'Fourier', algParams = None):
                CGHGenerator.requireTorch("The ADAM algorithm")
                if(algParams is None):
                        algParams = {}
                phase_ip_init = self.resolveInitialPhaseMap(initialPhase)

                # initial MSE
                msel = np.zeros((numIter, 1))

                # peak signal to noise ratio
                psnrl = np.zeros((numIter, 1))

                # define hyperparameters
                learning_rate = float(algParams.get("learningRate", 0.32))
                lossMode = self._resolveLossMode(algParams.get("lossMode", getattr(self, "lossMode", "auto")))

                phase_hp = phase_ip_init 
                loss = 0

                best_loss = 1e10
                image_recon = np.zeros(self.imTarget.shape)

                # Define pytorch model and optimizer
                phase_hp = torch.tensor(phase_hp, dtype=torch.float32, device=self.MLDevice)
                
                phase_hp.requires_grad = True
                best_phase = phase_hp.detach().clone()
                illuminationState = self._prepareTorchIlluminationOptimization(algParams, defaultLearningRate=learning_rate)
                best_illumination_param = (
                        illuminationState["param"].detach().clone()
                        if illuminationState["param"] is not None
                        else None
                )

                #Optimization Variables
                optvars = [{'params': phase_hp}]
                optvars.extend(illuminationState["optvars"])
                
                # Use ADAM for optimizer TODO:
                optimizer = torch.optim.Adam(optvars, lr=learning_rate)
                # Normalize the target image
                target_intensity = torch.tensor(self.imTarget, dtype=torch.float32, device=self.MLDevice)

                loss_weights = self._torchLossWeights(target_intensity, lossMode)

                # Loop ADAM
                for i in tqdm(range(0, numIter)):
                        optimizer.zero_grad() # Zero the gradients out

                        # Get PLM field
                        illumination_tensor = self._torchCurrentIllumination(illuminationState)
                        E_ip_inp = illumination_tensor * torch.exp(1j*phase_hp)

                        E_hp = self.torchProp("forward", E_ip_inp, propMethod)

                        # Get the absolute value of the hologram plane
                        E_hp_intensity = CGHGenerator.torchFieldToIntensity(E_hp)

                        # Optimize the scaled reconstruction directly, including the scale factor.
                        with torch.no_grad():
                                s = self._torchWeightedGain(E_hp_intensity, target_intensity, loss_weights)
                         
                        image_recon = s*E_hp_intensity
                        loss_val = self._torchWeightedMSE(image_recon, target_intensity, loss_weights)
                        if(illuminationState["smoothnessWeight"] > 0):
                                loss_val = loss_val + illuminationState["smoothnessWeight"] * CGHGenerator.torchIlluminationSmoothnessLoss(illumination_tensor)

                        # Compare reconstructed image with target image
                        msel[i] = loss_val.item() # Convert the loss tensor to a numpy array and store it in msel[i]
                        psnrl[i] = CGHGenerator.computePSNR(msel[i])

                        # Display the reconstructed image
                        if(self.show_images):
                                image_recon_np = image_recon.detach().cpu().numpy()
                                cv2.imshow("AWCGS Image" ,image_recon_np)
                                cv2.setWindowTitle("AWCGS Image", "Iteration: " + str(i+1) + " MSE: " + str(msel[i]) + " PSNR: " + str(psnrl[i]))
                                cv2.waitKey(0)
                                print(msel[i])
                                print(psnrl[i])
                        
                        loss_val.backward()
                        optimizer.step()
                
                        with torch.no_grad():
                                if loss_val.item() < best_loss:
                                        best_phase = phase_hp.detach().clone()
                                        if(illuminationState["param"] is not None):
                                                best_illumination_param = illuminationState["param"].detach().clone()
                                        best_loss = loss_val.item()


                # Save output
                metric_name = "Balanced MSE" if lossMode == "balanced" else "MSE"
                psnr_name = "Balanced PSNR" if lossMode == "balanced" else "PSNR"
                print("Best " + metric_name + ": " + str(np.min(msel)))
                print("Best " + psnr_name + ": " + str(np.max(psnrl)))
                self._finalizeTorchIllumination(illuminationState, best_illumination_param)
                phase_hp = best_phase.detach().cpu().numpy() # Detach the tensor from the computation graph and convert to numpy array
                phase_alg_cont = np.mod(phase_hp, 2*np.pi)
                self.CGH_output_cont = phase_alg_cont

        # Run ADAM with jointly optimized temporal subframes.
        def runADAMTemporal(self, nLevel = None, pLevel = None, temporalFrames = 1, initialPhase = 'Random', numIter = 1, propMethod = 'Fourier',
                            temporalSeedJitter = 0.10, algParams = None):
                CGHGenerator.requireTorch("The ADAMTemporal algorithm")
                if(algParams is None):
                        algParams = {}

                temporalFrames = int(max(1, temporalFrames))
                trainQuantized = bool(algParams.get("trainQuantized", False))
                lossMode = self._resolveLossMode(algParams.get("lossMode", getattr(self, "lossMode", "auto")))
                phase_ip_init = self.resolveInitialPhaseMap(initialPhase, allowTemporalStack=True)

                learning_rate = float(algParams.get("learningRate", 0.32))
                best_phase = None
                best_phase_output = None
                best_loss = float("inf")
                best_psnr = float("-inf")
                quant_levels = None
                if(trainQuantized):
                        if(nLevel is None or pLevel is None):
                                raise ValueError("Quantized temporal ADAM requires nLevel and pLevel.")
                        quant_levels = torch.tensor(
                                2*np.pi*np.asarray(pLevel[:nLevel], dtype=np.float32).reshape(1, 1, 1, -1),
                                dtype=torch.float32,
                                device=self.MLDevice
                        )

                phase_stack = self.buildTemporalInitialPhaseStack(
                        temporalFrames,
                        phase_ip_init,
                        initialPhase=initialPhase,
                        temporalSeedJitter=temporalSeedJitter
                )

                phase_hp = torch.tensor(phase_stack, dtype=torch.float32, device=self.MLDevice)
                phase_hp.requires_grad_(True)
                best_phase = phase_hp.detach().clone()
                illuminationState = self._prepareTorchIlluminationOptimization(algParams, defaultLearningRate=learning_rate)
                best_illumination_param = (
                        illuminationState["param"].detach().clone()
                        if illuminationState["param"] is not None
                        else None
                )

                optvars = [{'params': phase_hp}]
                optvars.extend(illuminationState["optvars"])
                optimizer = torch.optim.Adam(optvars, lr=learning_rate)
                target_intensity = torch.tensor(self.imTarget, dtype=torch.float32, device=self.MLDevice)
                loss_weights = self._torchLossWeights(target_intensity, lossMode)

                for i in tqdm(range(0, numIter)):
                        optimizer.zero_grad(set_to_none=True)

                        illumination_tensor = self._torchCurrentIllumination(illuminationState)
                        if(trainQuantized):
                                phase_for_field = CGHGenerator.torchNearestPhaseSTE(phase_hp, quant_levels)
                        else:
                                phase_for_field = phase_hp
                        E_ip_inp = illumination_tensor[None, :, :] * torch.exp(1j*phase_for_field)
                        E_hp = self.torchProp("forward", E_ip_inp, propMethod)

                        frame_intensity = CGHGenerator.torchFieldToIntensity(E_hp)
                        recon_intensity = torch.mean(frame_intensity, dim=0)
                        with torch.no_grad():
                                s = self._torchWeightedGain(recon_intensity, target_intensity, loss_weights)

                        image_recon = s * recon_intensity
                        mse_loss = self._torchWeightedMSE(image_recon, target_intensity, loss_weights)
                        loss_val = mse_loss
                        if(illuminationState["smoothnessWeight"] > 0):
                                loss_val = loss_val + illuminationState["smoothnessWeight"] * CGHGenerator.torchIlluminationSmoothnessLoss(illumination_tensor)

                        mse_val = float(mse_loss.detach().cpu())
                        psnr_val = CGHGenerator.computePSNR(mse_val)

                        if(self.show_images):
                                image_recon_np = image_recon.detach().cpu().numpy()
                                cv2.imshow("ADAM Temporal Image", image_recon_np)
                                cv2.setWindowTitle("ADAM Temporal Image", "Iteration: " + str(i+1) + " MSE: " + str(mse_val) + " PSNR: " + str(psnr_val))
                                cv2.waitKey(0)

                        loss_val.backward()
                        optimizer.step()

                        with torch.no_grad():
                                phase_hp.remainder_(2*np.pi)
                                if mse_val < best_loss:
                                        best_phase = phase_hp.detach().clone()
                                        if(trainQuantized):
                                                best_phase_output = phase_for_field.detach().clone()
                                        else:
                                                best_phase_output = None
                                        if(illuminationState["param"] is not None):
                                                best_illumination_param = illuminationState["param"].detach().clone()
                                        best_loss = mse_val
                                        best_psnr = psnr_val

                metric_name = "Balanced MSE" if lossMode == "balanced" else "MSE"
                psnr_name = "Balanced PSNR" if lossMode == "balanced" else "PSNR"
                print("Best temporal ADAM " + metric_name + ": " + str(best_loss))
                print("Best temporal ADAM " + psnr_name + ": " + str(best_psnr))
                self._finalizeTorchIllumination(illuminationState, best_illumination_param)

                if(best_phase_output is None):
                        if(trainQuantized):
                                best_phase_output = CGHGenerator.torchNearestPhase(best_phase, quant_levels).detach().clone()
                        else:
                                best_phase_output = best_phase.detach().clone()

                phase_stack = np.asarray(best_phase_output.detach().cpu().numpy(), dtype=np.float32)
                if(temporalFrames > 1):
                        self.CGH_output_cont_subframes = np.mod(phase_stack, 2*np.pi)
                        self.CGH_output_cont = self.CGH_output_cont_subframes[0].copy()
                else:
                        self.CGH_output_cont_subframes = None
                        self.CGH_output_cont = np.mod(phase_stack[0], 2*np.pi)

        # Run ADAM with Gumbel-Softmax Quantization
        def runADAMwGS(self, nLevel, pLevel, initialPhase = 'Random', numIter = 1, propMethod = 'Fourier', algParams = None):
                CGHGenerator.requireTorch("The ADAMWGS algorithm")
                if(algParams is None):
                        algParams = {}
                phase_ip_init = self.resolveInitialPhaseMap(initialPhase)

                # initial MSE
                msel = np.zeros((numIter, 1))

                # peak signal to noise ratio
                psnrl = np.zeros((numIter, 1))

                # define hyperparameters
                learning_rate = 0.32 # learning rate

                phase_hp = phase_ip_init
                best_quantized_phase = None

                best_loss = 1e10
                image_recon = np.zeros(self.imTarget.shape)

                # Define pytorch model and optimizer
                phase_hp = torch.tensor(phase_hp, dtype=torch.float32, device=self.MLDevice)
                phase_hp.requires_grad_(True) 
                best_phase = phase_hp.detach().clone()
                illuminationState = self._prepareTorchIlluminationOptimization(algParams, defaultLearningRate=learning_rate)
                best_illumination_param = (
                        illuminationState["param"].detach().clone()
                        if illuminationState["param"] is not None
                        else None
                )

                # Create logits
                levels = pLevel[:nLevel].reshape(1, 1, -1)
                logits =  np.tile(levels, (self.imTarget.shape[0], self.imTarget.shape[1], 1))
                logits = torch.tensor(logits, device = self.MLDevice, dtype=torch.float32)

                # Define loss functions
                loss_fun = nn.MSELoss()
                
                # Define temperatures for Gumbel-softmax
                tauInitial = torch.tensor(5.5, dtype=torch.float32, device=self.MLDevice)
                tauMin = torch.tensor(2.0, dtype=torch.float32, device=self.MLDevice)

                # Define quantization method
                quantizeMethod = self.torchquantize(lut=logits, levels=nLevel,tau=5.5, quantizeMethod="gumbelsoftmax")

                #Optimization Variables
                optvars = [{'params': phase_hp}]#, {'params': quantizeMethod.tau}]
                optvars.extend(illuminationState["optvars"])
                
                # Use ADAM for optimizer
                optimizer = torch.optim.Adam(optvars, lr=learning_rate)
                
                # Normalize the target image
                target = torch.tensor(self.imTarget, dtype=torch.float32, device=self.MLDevice) 

                target_intensity = target

                # Loop ADAM w Gumbel-Softmax Quantization
                for i in tqdm(range(0, numIter)):

                        optimizer.zero_grad() # Zero the gradients out

                        # Quantize the phase
                        quantized_phase = quantizeMethod(phase_hp)

                        # Get PLM field
                        illumination_tensor = self._torchCurrentIllumination(illuminationState)
                        E_ip_inp = illumination_tensor * torch.exp(1j*quantized_phase)

                        E_hp = self.torchProp("forward", E_ip_inp, propMethod)

                        # Get the absolute value of the hologram plane
                        E_hp_intensity = CGHGenerator.torchFieldToIntensity(E_hp)

                        s, _ = CGHGenerator.torchScaleIntensityToTarget(E_hp_intensity, target_intensity)

                        image_recon = s*E_hp_intensity
                        loss_val = loss_fun(image_recon, target_intensity)
                        if(illuminationState["smoothnessWeight"] > 0):
                                loss_val = loss_val + illuminationState["smoothnessWeight"] * CGHGenerator.torchIlluminationSmoothnessLoss(illumination_tensor)

                        # Compare reconstructed image with target image
                        msel[i] = ((image_recon - target_intensity)**2).mean().item() # Convert the loss tensor to a numpy array and store it in msel[i]
                        psnrl[i] = CGHGenerator.computePSNR(msel[i])

                        # Display the reconstructed image
                        if(self.show_images):
                                image_recon_np = image_recon.detach().cpu().numpy()
                                cv2.imshow("AWCGS Image" ,image_recon_np)
                                cv2.setWindowTitle("AWCGS Image", "Iteration: " + str(i+1) + " MSE: " + str(msel[i]) + " PSNR: " + str(psnrl[i]))
                                cv2.waitKey(0)
                                print(msel[i])
                                print(psnrl[i])
                        
                        loss_val.backward()
                        optimizer.step()

                        annealRate = torch.tensor(i/max(numIter, 1), dtype=torch.float32, device=self.MLDevice)
                        quantizeMethod.anneal_temperature(annealRate, tauInitial=tauInitial, tauMin=tauMin)
                
                        with torch.no_grad():
                                if loss_val.item() < best_loss:
                                        best_phase = phase_hp.detach().clone()
                                        best_quantized_phase = quantized_phase.detach().clone()
                                        if(illuminationState["param"] is not None):
                                                best_illumination_param = illuminationState["param"].detach().clone()
                                        best_loss = loss_val.item()
                # Save output
                print("Best MSE: " + str(np.min(msel)))
                print("Best PSNR: " + str(np.max(psnrl)))
                if(best_quantized_phase is None):
                        best_quantized_phase = quantizeMethod(best_phase).detach().clone()
                self._finalizeTorchIllumination(illuminationState, best_illumination_param)
                phase_hp = np.squeeze(best_quantized_phase.detach().cpu().numpy()) # Save the best quantized hologram that was actually evaluated in-loop.
                phase_alg_cont = np.mod(phase_hp, 2*np.pi)
                self.CGH_output_cont = phase_alg_cont

        # Run ADAMWGS with jointly optimized temporal subframes.
        # The subframes are optimized together against the fused temporal intensity,
        # so they complement each other rather than being solved independently.
        def runADAMWGSTemporal(self, nLevel, pLevel, temporalFrames = 1, initialPhase = 'Random', numIter = 1, propMethod = 'Fourier',
                               temporalSeedJitter = 0.10, algParams = None):
                CGHGenerator.requireTorch("The ADAMWGSTemporal algorithm")
                if(algParams is None):
                        algParams = {}

                temporalFrames = int(max(1, temporalFrames))
                phase_ip_init = self.resolveInitialPhaseMap(initialPhase, allowTemporalStack=True)

                learning_rate = float(algParams.get("learningRate", 0.32))
                best_phase = None
                best_phase_output = None
                best_loss = float("inf")
                best_psnr = float("-inf")

                phase_stack = self.buildTemporalInitialPhaseStack(
                        temporalFrames,
                        phase_ip_init,
                        initialPhase=initialPhase,
                        temporalSeedJitter=temporalSeedJitter
                )

                phase_hp = torch.tensor(phase_stack, dtype=torch.float32, device=self.MLDevice)
                phase_hp.requires_grad_(True)
                best_phase = phase_hp.detach().clone()
                illuminationState = self._prepareTorchIlluminationOptimization(algParams, defaultLearningRate=learning_rate)
                best_illumination_param = (
                        illuminationState["param"].detach().clone()
                        if illuminationState["param"] is not None
                        else None
                )

                levels = pLevel[:nLevel].reshape(1, 1, -1)
                logits = torch.tensor(levels, device=self.MLDevice, dtype=torch.float32)

                loss_fun = nn.MSELoss()
                tauInitial = torch.tensor(5.5, dtype=torch.float32, device=self.MLDevice)
                tauMin = torch.tensor(2.0, dtype=torch.float32, device=self.MLDevice)
                quantizeMethod = self.torchquantize(lut=logits, levels=nLevel, tau=5.5, quantizeMethod="gumbelsoftmax")
                optvars = [{'params': phase_hp}]
                optvars.extend(illuminationState["optvars"])
                optimizer = torch.optim.Adam(optvars, lr=learning_rate)
                target_intensity = torch.tensor(self.imTarget, dtype=torch.float32, device=self.MLDevice)

                for i in tqdm(range(0, numIter)):
                        optimizer.zero_grad(set_to_none=True)

                        quantized_phase = quantizeMethod(phase_hp)
                        illumination_tensor = self._torchCurrentIllumination(illuminationState)
                        E_ip_inp = illumination_tensor[None, :, :] * torch.exp(1j*quantized_phase)
                        E_hp = self.torchProp("forward", E_ip_inp, propMethod)

                        frame_intensity = CGHGenerator.torchFieldToIntensity(E_hp)
                        recon_intensity = torch.mean(frame_intensity, dim=0)

                        s, _ = CGHGenerator.torchScaleIntensityToTarget(recon_intensity, target_intensity)

                        image_recon = s*recon_intensity
                        mse_loss = loss_fun(image_recon, target_intensity)
                        loss_val = mse_loss
                        if(illuminationState["smoothnessWeight"] > 0):
                                loss_val = loss_val + illuminationState["smoothnessWeight"] * CGHGenerator.torchIlluminationSmoothnessLoss(illumination_tensor)

                        mse_val = float(mse_loss.detach().cpu())
                        psnr_val = CGHGenerator.computePSNR(mse_val)

                        with torch.no_grad():
                                deterministic_quantized_phase = quantizeMethod.deterministic(phase_hp)
                                deterministic_E_ip_inp = illumination_tensor[None, :, :] * torch.exp(1j*deterministic_quantized_phase)
                                deterministic_E_hp = self.torchProp("forward", deterministic_E_ip_inp, propMethod)
                                deterministic_frame_intensity = CGHGenerator.torchFieldToIntensity(deterministic_E_hp)
                                deterministic_recon_intensity = torch.mean(deterministic_frame_intensity, dim=0)
                                deterministic_s, _ = CGHGenerator.torchScaleIntensityToTarget(
                                        deterministic_recon_intensity,
                                        target_intensity
                                )
                                deterministic_image_recon = deterministic_s * deterministic_recon_intensity
                                deterministic_mse_loss = loss_fun(deterministic_image_recon, target_intensity)
                                deterministic_mse_val = float(deterministic_mse_loss.detach().cpu())
                                deterministic_psnr_val = CGHGenerator.computePSNR(deterministic_mse_val)

                        if(self.show_images):
                                image_recon_np = image_recon.detach().cpu().numpy()
                                cv2.imshow("ADAMWGS Temporal Image", image_recon_np)
                                cv2.setWindowTitle("ADAMWGS Temporal Image", "Iteration: " + str(i+1) + " MSE: " + str(mse_val) + " PSNR: " + str(psnr_val))
                                cv2.waitKey(0)

                        loss_val.backward()
                        optimizer.step()

                        annealRate = torch.tensor(i/max(numIter, 1), dtype=torch.float32, device=self.MLDevice)
                        quantizeMethod.anneal_temperature(annealRate, tauInitial=tauInitial, tauMin=tauMin)

                        with torch.no_grad():
                                if deterministic_mse_val < best_loss:
                                        best_phase = phase_hp.detach().clone()
                                        best_phase_output = deterministic_quantized_phase.detach().clone()
                                        if(illuminationState["param"] is not None):
                                                best_illumination_param = illuminationState["param"].detach().clone()
                                        best_loss = deterministic_mse_val
                                        best_psnr = deterministic_psnr_val

                print("Best temporal in-loop discrete MSE: " + str(best_loss))
                print("Best temporal in-loop discrete PSNR: " + str(best_psnr))
                self._finalizeTorchIllumination(illuminationState, best_illumination_param)

                if(best_phase_output is None):
                        best_phase_output = quantizeMethod(best_phase).detach().clone()

                phase_stack = np.asarray(best_phase_output.detach().cpu().numpy(), dtype=np.float32)

                if(temporalFrames > 1):
                        self.CGH_output_cont_subframes = np.mod(phase_stack, 2*np.pi)
                        self.CGH_output_cont = self.CGH_output_cont_subframes[0].copy()
                else:
                        self.CGH_output_cont_subframes = None
                        self.CGH_output_cont = np.mod(phase_stack[0], 2*np.pi)

        # Run ADAMWGS with jointly optimized visible-wavelength temporal subframes.
        # Each temporal slot uses its own LUT wavelength and optional illumination wavelength,
        # so the optimizer can exploit multiple visible phase tables together.
        def runADAMWGSVisibleTemporal(self, DeviceDictionary, frameLUTWavelengths = None, frameIlluminationWavelengths = None,
                                      initialPhase = 'Random', numIter = 1, propMethod = 'Fourier', temporalSeedJitter = 0.10,
                                      algParams = None):
                CGHGenerator.requireTorch("The ADAMWGSVISIBLETEMPORAL algorithm")
                if(algParams is None):
                        algParams = {}

                if(propMethod.upper() != "FOURIER"):
                        raise ValueError("ADAMWGSVISIBLETEMPORAL currently supports Fourier propagation only.")
                if(frameLUTWavelengths is None):
                        raise ValueError("ADAMWGSVISIBLETEMPORAL requires frameLUTWavelengths, for example [650, 532].")

                frame_setup = CGHGenerator.buildVisibleTemporalFrameDevices(
                        DeviceDictionary,
                        frameLUTWavelengths=frameLUTWavelengths,
                        frameIlluminationWavelengths=frameIlluminationWavelengths
                )
                frame_devices = frame_setup["frame_devices"]
                frame_pLevels = frame_setup["frame_pLevels"]
                frame_lut_m = frame_setup["frame_lut_m"]
                frame_illum_m = frame_setup["frame_illum_m"]
                temporalFrames = len(frame_pLevels)

                phase_ip_init = self.resolveInitialPhaseMap(initialPhase, allowTemporalStack=True)

                learning_rate = 0.32
                best_phase = None
                best_quantized_stack = None
                best_loss = float("inf")
                best_psnr = float("-inf")

                phase_stack = self.buildTemporalInitialPhaseStack(
                        temporalFrames,
                        phase_ip_init,
                        initialPhase=initialPhase,
                        temporalSeedJitter=temporalSeedJitter
                )

                phase_hp = torch.tensor(phase_stack, dtype=torch.float32, device=self.MLDevice)
                phase_hp.requires_grad_(True)
                best_phase = phase_hp.detach().clone()
                illuminationState = self._prepareTorchIlluminationOptimization(algParams, defaultLearningRate=learning_rate)
                best_illumination_param = (
                        illuminationState["param"].detach().clone()
                        if illuminationState["param"] is not None
                        else None
                )

                quantizers = []
                for frame_idx in range(temporalFrames):
                        levels = np.asarray(frame_pLevels[frame_idx][:frame_devices[frame_idx]["nLevel"]], dtype=np.float32).reshape(1, 1, -1)
                        levels = torch.tensor(levels, device=self.MLDevice, dtype=torch.float32)
                        quantizers.append(self.torchquantize(lut=levels, levels=frame_devices[frame_idx]["nLevel"], tau=5.5, quantizeMethod="gumbelsoftmax"))

                loss_fun = nn.MSELoss()
                tauInitial = torch.tensor(5.5, dtype=torch.float32, device=self.MLDevice)
                tauMin = torch.tensor(2.0, dtype=torch.float32, device=self.MLDevice)
                optvars = [{'params': phase_hp}]
                optvars.extend(illuminationState["optvars"])
                optimizer = torch.optim.Adam(optvars, lr=learning_rate)
                target_intensity = torch.tensor(self.imTarget, dtype=torch.float32, device=self.MLDevice)

                for i in tqdm(range(0, numIter)):
                        optimizer.zero_grad(set_to_none=True)

                        quantized_frames = []
                        for frame_idx in range(temporalFrames):
                                quantized_frames.append(quantizers[frame_idx](phase_hp[frame_idx]))
                        quantized_phase = torch.stack(quantized_frames, dim=0)

                        illumination_tensor = self._torchCurrentIllumination(illuminationState)
                        E_ip_inp = illumination_tensor[None, :, :] * torch.exp(1j*quantized_phase)
                        E_hp = self.torchProp("forward", E_ip_inp, propMethod)

                        frame_intensity = CGHGenerator.torchFieldToIntensity(E_hp)
                        recon_intensity = torch.mean(frame_intensity, dim=0)

                        s, _ = CGHGenerator.torchScaleIntensityToTarget(recon_intensity, target_intensity)

                        image_recon = s*recon_intensity
                        mse_loss = loss_fun(image_recon, target_intensity)
                        loss_val = mse_loss
                        if(illuminationState["smoothnessWeight"] > 0):
                                loss_val = loss_val + illuminationState["smoothnessWeight"] * CGHGenerator.torchIlluminationSmoothnessLoss(illumination_tensor)

                        mse_val = float(mse_loss.detach().cpu())
                        psnr_val = CGHGenerator.computePSNR(mse_val)

                        if(self.show_images):
                                image_recon_np = image_recon.detach().cpu().numpy()
                                cv2.imshow("ADAMWGS Visible Temporal Image", image_recon_np)
                                cv2.setWindowTitle(
                                        "ADAMWGS Visible Temporal Image",
                                        "Iteration: " + str(i+1) + " MSE: " + str(mse_val) + " PSNR: " + str(psnr_val)
                                )
                                cv2.waitKey(0)

                        loss_val.backward()
                        optimizer.step()

                        annealRate = torch.tensor(i/max(numIter, 1), dtype=torch.float32, device=self.MLDevice)
                        for quantizer in quantizers:
                                quantizer.anneal_temperature(annealRate, tauInitial=tauInitial, tauMin=tauMin)

                        with torch.no_grad():
                                if mse_val < best_loss:
                                        best_phase = phase_hp.detach().clone()
                                        best_quantized_stack = quantized_phase.detach().clone()
                                        if(illuminationState["param"] is not None):
                                                best_illumination_param = illuminationState["param"].detach().clone()
                                        best_loss = mse_val
                                        best_psnr = psnr_val

                print("Best visible temporal in-loop discrete MSE: " + str(best_loss))
                print("Best visible temporal in-loop discrete PSNR: " + str(best_psnr))
                self._finalizeTorchIllumination(illuminationState, best_illumination_param)

                if(best_quantized_stack is None):
                        best_quantized_stack = torch.stack(
                                [quantizers[frame_idx](best_phase[frame_idx]) for frame_idx in range(temporalFrames)],
                                dim=0
                        ).detach().clone()

                phase_stack = np.asarray(best_quantized_stack.detach().cpu().numpy(), dtype=np.float32)
                phase_stack = np.mod(phase_stack, 2*np.pi)

                phase_maps = np.zeros_like(phase_stack, dtype=np.float32)
                state_maps = np.zeros_like(phase_stack, dtype=np.float32)
                for frame_idx in range(temporalFrames):
                        phase_disc, state_disc = self.discretePhase(
                                phase_stack[frame_idx],
                                frame_devices[frame_idx]["nLevel"],
                                frame_pLevels[frame_idx]
                        )
                        phase_maps[frame_idx] = (2*np.pi*phase_disc).astype(np.float32)
                        state_maps[frame_idx] = state_disc.astype(np.float32)

                self.temporalFrameLUTWavelengths = np.asarray(frame_lut_m, dtype=np.float64)
                self.temporalFrameIlluminationWavelengths = np.asarray(frame_illum_m, dtype=np.float64)

                print(
                        "Temporal frame LUT wavelengths (nm): "
                        + str(np.round(self.temporalFrameLUTWavelengths * 1e9, 3))
                )

                self.CGH_output_cont = phase_maps[0].copy()
                self.CGH_output_disc = phase_maps[0].copy()
                self.CGH_output_phase_disc = phase_maps[0].copy()
                self.CGH_output_state_disc = state_maps[0].copy()
                self.CGH_output_cont_subframes = phase_maps.copy()
                self.CGH_output_phase_disc_subframes = phase_maps.copy()
                self.CGH_output_state_disc_subframes = state_maps.copy()
                self.CGH_effective_field_cont = None
                self.CGH_effective_field_disc = None
                self.CGH_effective_field_cont_subframes = None
                self.CGH_effective_field_disc_subframes = None

        def createVisibleSummedCGH(self, DeviceDictionary,
                                    filename = "", colorChannel = 0,
                                    FlipUD = False, FlipLR = False, InvertTarget = False,
                                    groupLUTWavelengths = (532, 532), temporalFrames = 1,
                                    frameIlluminationWavelengths = None,
                                    numIter = 1, initialPhase = 'Random', propMethod = 'Fourier',
                                    ShiftFOV = True, showImages = False,
                                    temporalSeedJitter = 0.10,
                                    independentGroupSeeds = False):
                CGHGenerator.requireTorch("The visible multi-wavelength summed-intensity CGH generator")

                if(propMethod.upper() != "FOURIER"):
                        raise ValueError("createVisibleSummedCGH currently supports Fourier propagation only.")

                self.show_images = showImages
                self.bitPlanes = 1

                self.CGH_mapped = None
                self.CGH_phase = None
                self.imRecovered_cont = None
                self.imRecovered_disc = None
                self.optimizedIllumination = None
                self.illuminationWasOptimized = False
                self.CGH_output_cont = None
                self.CGH_output_cont_subframes = None
                self.CGH_output_disc = None
                self.CGH_output_phase_disc = None
                self.CGH_output_state_disc = None
                self.CGH_output_phase_disc_subframes = None
                self.CGH_output_state_disc_subframes = None
                self.CGH_effective_field_cont = None
                self.CGH_effective_field_disc = None
                self.CGH_effective_field_cont_subframes = None
                self.CGH_effective_field_disc_subframes = None
                self.temporalFrameLUTWavelengths = None
                self.temporalFrameIlluminationWavelengths = None
                self.CGH_is_sequence = False
                self.imRecovered_cont_native = None
                self.imRecovered_disc_native = None
                self.softWTAGroupLUTWavelengths = None
                self.softWTAGroupTemporalFrames = None
                self.softWTASelectorWeights = None
                self.softWTAHardWinnerMap = None
                self.softWTASoftFusedNative = None
                self.softWTAHardFusedNative = None
                self.softWTAChosenFusionMode = None
                self.softWTAChosenFusionMSE = None
                self.softWTAChosenFusionPSNR = None
                self.softWTAGroupReconsNative = None
                self.softWTAGroupOutputs = None
                self.outputShape = (DeviceDictionary["h"], DeviceDictionary["w"])

                self.loadTarget(filename, colorChannel)
                self.resizeTarget(
                        self.lambda_m,
                        DeviceDictionary["h"],
                        DeviceDictionary["w"],
                        DeviceDictionary["pitchW"],
                        DeviceDictionary["pitchH"],
                        ShiftFOV
                )
                self.updateTarget(FlipUD, FlipLR, InvertTarget)
                self.imTarget_recon = self.imTarget.copy()

                wavelength_groups_m = CGHGenerator.normalizeWavelengthArray(groupLUTWavelengths, "groupLUTWavelengths")
                num_groups = int(wavelength_groups_m.size)
                if(num_groups < 2):
                        raise ValueError("createVisibleSummedCGH requires at least two wavelength groups.")

                temporal_frames_per_group = CGHGenerator.normalizeTemporalFramesPerGroup(
                        temporalFrames,
                        num_groups,
                        name="temporalFrames"
                )

                if(frameIlluminationWavelengths is None):
                        illumination_groups_m = wavelength_groups_m.copy()
                else:
                        illumination_groups_m = CGHGenerator.normalizeWavelengthArray(
                                frameIlluminationWavelengths,
                                "frameIlluminationWavelengths"
                        )
                        if(int(illumination_groups_m.size) != num_groups):
                                raise ValueError("frameIlluminationWavelengths must match groupLUTWavelengths in length.")

                if(DeviceDictionary["device"] not in ("0.47", "0.67", "0.98")):
                        raise ValueError(
                                "Visible multi-wavelength optimization currently supports the visible empirical devices "
                                "0.47, 0.67, and 0.98."
                        )

                group_devices = []
                group_pLevels = []
                for lut_m, illum_m in zip(wavelength_groups_m, illumination_groups_m):
                        group_device = DeviceLibrary().defineDevice(
                                DeviceDictionary["device"],
                                discScheme=DeviceDictionary.get("discScheme", "Empirical"),
                                lutWavelength=lut_m
                        )
                        pLevel = np.asarray(group_device["pLevel"], dtype=np.float64)
                        if(pLevel.ndim != 1):
                                raise ValueError("Visible multi-wavelength optimization expects a 1D phase LUT per wavelength group.")

                        scale = float(lut_m / illum_m)
                        effective_pLevel = np.remainder(pLevel * scale, 1.0)
                        scaled_endpoint = float(pLevel[-1] * scale)
                        if(np.isclose(scaled_endpoint, 1.0, atol=1e-9)):
                                effective_pLevel[-1] = 1.0
                        else:
                                effective_pLevel[-1] = np.remainder(scaled_endpoint, 1.0)

                        group_devices.append(group_device)
                        group_pLevels.append(effective_pLevel)

                initialPhaseMode = str(initialPhase).upper()
                phase_ip_init = self.resolveInitialPhaseMap(initialPhaseMode, allowTemporalStack=True)

                group_slices = []
                group_phase_stacks = []
                total_frames = 0
                for group_idx in range(num_groups):
                        if(independentGroupSeeds and initialPhaseMode == "RANDOM"):
                                group_base_phase = 2*np.pi * np.random.rand(self.imTarget.shape[0], self.imTarget.shape[1])
                        else:
                                group_base_phase = phase_ip_init

                        group_phase_stack = self.buildTemporalInitialPhaseStack(
                                int(temporal_frames_per_group[group_idx]),
                                group_base_phase,
                                initialPhase=initialPhaseMode,
                                temporalSeedJitter=temporalSeedJitter
                        )
                        group_phase_stacks.append(group_phase_stack)
                        group_slices.append(slice(total_frames, total_frames + group_phase_stack.shape[0]))
                        total_frames = total_frames + group_phase_stack.shape[0]

                phase_stack = np.concatenate(group_phase_stacks, axis=0).astype(np.float32)
                phase_hp = torch.tensor(phase_stack, dtype=torch.float32, device=self.MLDevice)
                phase_hp.requires_grad_(True)

                quantizers = []
                for group_idx in range(num_groups):
                        levels = np.asarray(
                                group_pLevels[group_idx][:group_devices[group_idx]["nLevel"]],
                                dtype=np.float32
                        ).reshape(1, 1, -1)
                        levels = torch.tensor(levels, device=self.MLDevice, dtype=torch.float32)
                        for _ in range(int(temporal_frames_per_group[group_idx])):
                                quantizers.append(
                                        self.torchquantize(
                                                lut=levels,
                                                levels=group_devices[group_idx]["nLevel"],
                                                tau=5.5,
                                                quantizeMethod="gumbelsoftmax"
                                        )
                                )

                learning_rate = 0.32
                tauInitial = torch.tensor(5.5, dtype=torch.float32, device=self.MLDevice)
                tauMin = torch.tensor(2.0, dtype=torch.float32, device=self.MLDevice)
                optimizer = torch.optim.Adam([{'params': phase_hp}], lr=learning_rate)
                loss_fun = nn.MSELoss()
                target_intensity = torch.tensor(self.imTarget, dtype=torch.float32, device=self.MLDevice)

                best_loss = float("inf")
                best_quantized_stack = None
                best_fused_mse = float("inf")
                best_fused_psnr = float("-inf")

                for i in tqdm(range(0, numIter)):
                        optimizer.zero_grad(set_to_none=True)

                        quantized_frames = []
                        for frame_idx in range(total_frames):
                                quantized_frames.append(quantizers[frame_idx](phase_hp[frame_idx]))
                        quantized_phase = torch.stack(quantized_frames, dim=0)

                        E_ip_inp = torch.exp(1j*quantized_phase)
                        E_hp = self.torchProp("forward", E_ip_inp, propMethod)
                        frame_intensity = CGHGenerator.torchFieldToIntensity(E_hp)

                        # Sum every displayed frame equally so the loss matches exposure-integrated temporal fusion.
                        fused_recon = torch.sum(frame_intensity, dim=0)

                        s_fused, _ = CGHGenerator.torchScaleIntensityToTarget(fused_recon, target_intensity)
                        fused_recon = s_fused * fused_recon

                        fused_mse = loss_fun(fused_recon, target_intensity)
                        loss_val = fused_mse

                        fused_mse_val = float(fused_mse.detach().cpu())
                        fused_psnr_val = CGHGenerator.computePSNR(fused_mse_val)

                        if(self.show_images):
                                image_recon_np = fused_recon.detach().cpu().numpy()
                                cv2.imshow("Visible Summed Multi-Wavelength Image", image_recon_np)
                                cv2.setWindowTitle(
                                        "Visible Summed Multi-Wavelength Image",
                                        "Iteration: " + str(i+1)
                                        + " Summed MSE: " + str(fused_mse_val)
                                        + " Summed PSNR: " + str(fused_psnr_val)
                                )
                                cv2.waitKey(0)

                        loss_val.backward()
                        optimizer.step()

                        annealRate = torch.tensor(i/max(numIter, 1), dtype=torch.float32, device=self.MLDevice)
                        for quantizer in quantizers:
                                quantizer.anneal_temperature(annealRate, tauInitial=tauInitial, tauMin=tauMin)

                        with torch.no_grad():
                                if(fused_mse_val < best_fused_mse):
                                        best_fused_mse = fused_mse_val
                                        best_fused_psnr = fused_psnr_val

                                if(fused_mse_val < best_loss):
                                        best_loss = fused_mse_val
                                        best_quantized_stack = quantized_phase.detach().clone()
                                        best_fused_mse = fused_mse_val
                                        best_fused_psnr = fused_psnr_val

                if(best_quantized_stack is None):
                        best_quantized_stack = torch.stack(
                                [quantizers[frame_idx](phase_hp[frame_idx]) for frame_idx in range(total_frames)],
                                dim=0
                        ).detach().clone()

                print("Best summed checkpoint MSE: " + str(best_loss))
                print("Best summed checkpoint PSNR: " + str(best_fused_psnr))

                quantized_stack_np = np.mod(
                        np.asarray(best_quantized_stack.detach().cpu().numpy(), dtype=np.float32),
                        2*np.pi
                )

                target_ref = self.getReconstructionTarget(self.imTarget.shape)
                group_power_sum_stack_np = []
                group_recon_stack_np = []
                selector_weights_np = None
                hard_winner_map = None
                soft_fused_np = None
                hard_fused_np = None
                group_outputs = []

                for group_idx, group_slice in enumerate(group_slices):
                        phase_maps = []
                        state_maps = []
                        mapped_frames = []
                        frame_powers = []

                        for frame_idx in range(group_slice.start, group_slice.stop):
                                phase_disc, state_disc = self.discretePhase(
                                        quantized_stack_np[frame_idx],
                                        group_devices[group_idx]["nLevel"],
                                        group_pLevels[group_idx]
                                )
                                phase_map = (2*np.pi*phase_disc).astype(np.float32)
                                state_map = state_disc.astype(np.float32)
                                mapped_frame = self.deviceLibary.formatPLM(group_devices[group_idx], state_map)

                                E_hp = np.exp(1j*phase_map)
                                E_ip = self.prop("forward", E_hp, propMethod)
                                frame_powers.append(CGHGenerator.fieldToIntensity(E_ip))

                                phase_maps.append(phase_map)
                                state_maps.append(state_map)
                                mapped_frames.append(mapped_frame)

                        frame_powers_np = np.asarray(frame_powers, dtype=np.float64)
                        group_power = np.sum(frame_powers_np, axis=0)
                        group_power_sum_stack_np.append(group_power)
                        _, group_recon = CGHGenerator.scaleIntensityToTarget(group_power, target_ref)
                        group_recon_stack_np.append(group_recon)

                        mapped_stack = np.stack(mapped_frames, axis=2)
                        group_outputs.append({
                                "lut_wavelength_m": float(wavelength_groups_m[group_idx]),
                                "illumination_wavelength_m": float(illumination_groups_m[group_idx]),
                                "temporal_frames": int(temporal_frames_per_group[group_idx]),
                                "phase_maps": np.asarray(phase_maps, dtype=np.float32),
                                "state_maps": np.asarray(state_maps, dtype=np.float32),
                                "mapped_frames": mapped_stack,
                                "reconstruction_native": group_recon.copy(),
                        })

                group_power_sum_stack_np = np.asarray(group_power_sum_stack_np, dtype=np.float64)
                group_recon_stack_np = np.asarray(group_recon_stack_np, dtype=np.float64)
                selector_weights_np = CGHGenerator.normalizeIntensityContributions(group_power_sum_stack_np)
                hard_winner_map = np.argmax(selector_weights_np, axis=0).astype(np.int32)
                soft_fused_np = np.sum(group_power_sum_stack_np, axis=0)
                _, soft_fused_np = CGHGenerator.scaleIntensityToTarget(soft_fused_np, target_ref)
                hard_fused_np = np.take_along_axis(group_recon_stack_np, hard_winner_map[None, :, :], axis=0)[0]
                _, hard_fused_np = CGHGenerator.scaleIntensityToTarget(hard_fused_np, target_ref)

                chosen_fused_np = soft_fused_np.copy()
                chosen_mode = "sum"
                chosen_mse = CGHGenerator.computeMSE(chosen_fused_np, target_ref)
                chosen_psnr = CGHGenerator.computePSNR(chosen_mse)

                if(ShiftFOV):
                        shift = (int(hard_fused_np.shape[0] / 2), int(hard_fused_np.shape[1] / 2))
                        group_recon_stack_np = np.roll(group_recon_stack_np, shift=(0, shift[0], shift[1]), axis=(0, 1, 2))
                        selector_weights_np = np.roll(selector_weights_np, shift=(0, shift[0], shift[1]), axis=(0, 1, 2))
                        hard_winner_map = np.roll(hard_winner_map, shift=shift, axis=(0, 1))
                        soft_fused_np = np.roll(soft_fused_np, shift=shift, axis=(0, 1))
                        hard_fused_np = np.roll(hard_fused_np, shift=shift, axis=(0, 1))
                        chosen_fused_np = np.roll(chosen_fused_np, shift=shift, axis=(0, 1))
                        for group_output in group_outputs:
                                group_output["reconstruction_native"] = np.roll(
                                        group_output["reconstruction_native"],
                                        shift=shift,
                                        axis=(0, 1)
                                )

                self.softWTAGroupLUTWavelengths = wavelength_groups_m.copy()
                self.softWTAGroupTemporalFrames = temporal_frames_per_group.copy()
                self.softWTASelectorWeights = np.asarray(selector_weights_np, dtype=np.float64)
                self.softWTAHardWinnerMap = hard_winner_map.copy()
                self.softWTASoftFusedNative = np.asarray(soft_fused_np, dtype=np.float64)
                self.softWTAHardFusedNative = np.asarray(hard_fused_np, dtype=np.float64)
                self.softWTAChosenFusionMode = chosen_mode
                self.softWTAChosenFusionMSE = float(chosen_mse)
                self.softWTAChosenFusionPSNR = float(chosen_psnr)
                self.softWTAGroupReconsNative = np.asarray(group_recon_stack_np, dtype=np.float64)
                self.softWTAGroupOutputs = group_outputs

                self.imRecovered_cont_native = np.asarray(chosen_fused_np, dtype=np.float64)
                self.imRecovered_disc_native = self.imRecovered_cont_native.copy()

                if(self.imRecovered_cont_native.shape != self.outputShape):
                        self.imRecovered_cont = cv2.resize(
                                self.imRecovered_cont_native,
                                (self.outputShape[1], self.outputShape[0]),
                                interpolation=cv2.INTER_NEAREST
                        )
                        self.imRecovered_disc = self.imRecovered_cont.copy()
                else:
                        self.imRecovered_cont = self.imRecovered_cont_native.copy()
                        self.imRecovered_disc = self.imRecovered_cont.copy()

                if(len(group_outputs) > 0):
                        self.CGH_output_cont = group_outputs[0]["phase_maps"][0].copy()
                        self.CGH_output_phase_disc = group_outputs[0]["phase_maps"][0].copy()
                        self.CGH_output_state_disc = group_outputs[0]["state_maps"][0].copy()
                        self.CGH_output_disc = self.CGH_output_phase_disc.copy()
                        self.CGH_mapped = group_outputs[0]["mapped_frames"].copy()
                        self.CGH_phase = np.moveaxis(group_outputs[0]["phase_maps"].copy(), 0, 2)
                        self.CGH_is_sequence = bool(group_outputs[0]["mapped_frames"].ndim == 3 and group_outputs[0]["mapped_frames"].shape[2] > 1)

        # Run the mutlicolor algorithm. This needs to be ran for all 3 different bit planes (R,G,B)
        def runMultiColor(self, P = 3, T = 3, lambda_p = [532e-9, 532e-9, 532e-9], lambda_p_anch = 532e-9, initialPhase = 'Random', numIter = 1):
                CGHGenerator.requireTorch("The MULTICOLOR algorithm")
                phase_ip_init = self.resolveInitialPhaseMap(initialPhase)

                # initial MSE
                msel = np.zeros((numIter, 1))

                # peak signal to noise ratio
                psnrl = np.zeros((numIter, 1))

                # define hyperparameters
                learning_rate = 0.32 # learning rate for Adam algorithm

                phase_hp = phase_ip_init
                loss = 0

                best_loss = 1e10
                image_recon = np.zeros(self.imTarget.shape)

                # Define laser amplitude for the p-th laser and t-th subframe to start at 1 for each color
                l = np.ones((P,T)) # For RGB lets assume it would look like l[1 1 1] if there was only 1-bit plane but R,G,B or [[1 1 1] [1 1 1] [1 1 1]] for RGB 3-bit plane
                l = torch.tensor(l, dtype=torch.float32, device=self.MLDevice)
                l.requires_grad = True

                best_l = l.detach().cpu().numpy().copy()

                # Define pytorch model and optimizer
                phase_hp = torch.tensor(phase_hp, dtype=torch.float32, device=self.MLDevice)
                phase_hp.requires_grad = True
                best_phase = phase_hp.detach().clone()

                #Optimization Variables
                optvars = [{'params': phase_hp}]
                optvars.append({'params': l}) # Add the laser amplitude to the optimization variables
                
                # Use ADAM for optimizer
                optimizer = torch.optim.Adam(optvars, lr=learning_rate)
                # Normalize the target image
                target_intensity = torch.tensor(self.imTarget, dtype=torch.float32, device=self.MLDevice)

                # Define loss function
                loss = CGHGenerator.MultiColorLoss(P,T,lambda_p,lambda_p_anch).to(self.MLDevice)

                # Loop ADAM
                for i in tqdm(range(0, numIter)):
                        optimizer.zero_grad() # Zero the gradients out

                        loss_val = loss(l, phase_hp, target_intensity) # Compute the loss using the custom TriColorLoss function

                        # Compare reconstructed image with target image
                        msel[i] = loss_val.item() # Convert the loss tensor to a numpy array and store it in msel[i]
                        psnrl[i] = CGHGenerator.computePSNR(msel[i])

                        # Display the reconstructed image
                        if(self.show_images):
                                image_recon_np = self.reconstruct(phase_hp.detach().cpu().numpy())[0]
                                cv2.imshow("AWCGS Image" ,image_recon_np)
                                cv2.setWindowTitle("AWCGS Image", "Iteration: " + str(i+1) + " MSE: " + str(msel[i]) + " PSNR: " + str(psnrl[i]))
                                cv2.waitKey(0)
                                print(msel[i])
                                print(psnrl[i])
                        
                        loss_val.backward()
                        optimizer.step()
                
                        with torch.no_grad():
                                if loss_val.item() < best_loss:
                                        best_phase = phase_hp.detach().clone()
                                        best_loss = loss_val.item()
                                        best_iter = i + 1
                                        best_l = l.detach().cpu().numpy().copy()


                # Save output
                print("Best MSE: " + str(np.min(msel)))
                print("Best PSNR: " + str(np.max(psnrl)))
                print("Best Laser Amplitudes: ")
                print(best_l)
                phase_hp = best_phase.detach().cpu().numpy() # Detach the tensor from the computation graph and convert to numpy array
                phase_alg_cont = np.mod(phase_hp, 2*np.pi)
                self.CGH_output_cont = phase_alg_cont

        # Custom class for computing the Multi Color loss
        # P is the number of color primaries (R,G,B)
        # T is the number of bitplanes
        # lambda_p is the wavelength of each color primary (in meters)
        # lambda_p_anch is the wavelength of the anchor (green by default)
        class MultiColorLoss(nn.Module):
                def __init__(self,  P = 1, T = 1, lambda_p = [532e-9, 532e-9, 532e-9], lambda_p_anch = 532e-9):
                        super(CGHGenerator.MultiColorLoss, self).__init__()
                        self.P = P
                        self.T = T
                        self.lambda_p = lambda_p
                        self.lambda_p_anch = lambda_p_anch
                
                # Custom loss function
                def forward(self, l, phase_hp, target):
                        loss_image = 0
                        loss_laser = 0
                        loss_total = 0
                        # Compute image loss
                        for p in range(self.P):
                                E_hp_intensity = torch.zeros_like(phase_hp, dtype=torch.float32)
                                for t in range(self.T):
                                        # Get PLM field
                                        E_ip_inp = l[p,t]*torch.exp(1j*phase_hp*(self.lambda_p[p])/self.lambda_p_anch)

                                        E_hp = torch.fft.ifft2(E_ip_inp)

                                        # Image-plane intensity is the field magnitude squared.
                                        E_hp_intensity_loc = CGHGenerator.torchFieldToIntensity(E_hp)

                                        s, _ = CGHGenerator.torchScaleIntensityToTarget(E_hp_intensity_loc, target)
                                         
                                        E_hp_intensity = E_hp_intensity + s*E_hp_intensity_loc

                                loss_image = loss_image + torch.linalg.matrix_norm(E_hp_intensity - target)**2

                        # Computer laser loss
                        for p in range(self.P):
                                max_intensity = torch.max(target)
                                laser_intensities = 0

                                for t in range(self.T):
                                        laser_intensities = laser_intensities + l[p,t]**2

                                loss_laser = loss_laser + (laser_intensities - max_intensity)**2

                        loss_total = 3.0*loss_image + 0.05*loss_laser

                        return loss_total

        # Propogation of field for chosen algorithm
        def prop(self, direction, field_in, algorithm="FOURIER"):

                # Make direction case-insensitive
                direction = str(direction.upper())

                # Fourier propogation
                if(algorithm.upper() == "FOURIER"):
                        if(direction == "FORWARD"):
                                field_out = np.fft.ifft2(field_in)
                        elif(direction == "BACKWARD"):
                                field_out = np.fft.fft2(field_in)

                elif(algorithm.upper() == "ASM"):
                        if(self.pitchW is None or self.pitchH is None):
                                raise ValueError("ASM propagation requires pitchW and pitchH to be set on the CGHGenerator.")
                        if(getattr(self, "prop_dist", None) is None):
                                raise ValueError("ASM propagation requires prop_dist to be set on the CGHGenerator.")
                        # Calculate meshgrid of spatial frequencies 
                        kx_spacing = 1/self.imTarget.shape[1]/self.pitchW
                        kx_lst = np.array(range(0,self.imTarget.shape[1])) * kx_spacing
                        kx_lst = kx_lst - np.mean(kx_lst)

                        ky_spacing = 1/self.imTarget.shape[0]/self.pitchH
                        ky_lst = np.array(range(0,self.imTarget.shape[0])) * ky_spacing
                        ky_lst = ky_lst - np.mean(ky_lst)

                        mesh_kx, mesh_ky = np.meshgrid(kx_lst, ky_lst)
                        mesh_kx = 2*np.pi*mesh_kx
                        mesh_ky = 2*np.pi*mesh_ky
                        k = 2*np.pi/self.lambda_m

                        # Calculate transfer function
                        if(direction == "FORWARD"):
                                trans = np.exp(-1j * self.prop_dist * np.sqrt(k**2 - mesh_kx**2 - mesh_ky**2))
                                idx = np.sqrt(mesh_kx**2 + mesh_ky**2) > k
                                trans[idx] = 1

                                in_fft = np.fft.fft2(field_in)
                                trans_in_fft = trans*in_fft

                                field_out = np.fft.ifft2(trans_in_fft)

                        elif(direction == "BACKWARD"):
                                trans = np.exp(-1j * -self.prop_dist * np.sqrt(k**2 - mesh_kx**2 - mesh_ky**2))
                                idx = np.sqrt(mesh_kx**2 + mesh_ky**2) > k
                                trans[idx] = 1

                                in_fft = np.fft.fft2(field_in)
                                trans_in_fft = trans*in_fft

                                field_out = np.fft.ifft2(trans_in_fft)

                return field_out
        
        # Prop function using torch compatible methods
        def torchProp(self, direction, field_in, algorithm="FOURIER"):
                CGHGenerator.requireTorch("Torch-based propagation")

                # Make direction case-insensitive
                direction = str(direction.upper())

                # Fourier propogation
                if(algorithm.upper() == "FOURIER"):
                        if(direction == "FORWARD"):
                                field_out = torch.fft.ifft2(field_in)
                        elif(direction == "BACKWARD"):
                                field_out = torch.fft.fft2(field_in)
                
                return field_out

        ########################
        # Quantization methods #
        # and helper functions #
        ########################

        @staticmethod
        def torchNearestPhase(phase, quant_levels):
                wrapped_phase = torch.remainder(phase, 2*np.pi)
                levels = quant_levels.to(device=phase.device, dtype=phase.dtype)
                while(levels.dim() < wrapped_phase.dim() + 1):
                        levels = levels.unsqueeze(0)
                diff = wrapped_phase.unsqueeze(-1) - levels
                diff = torch.remainder(diff + np.pi, 2*np.pi) - np.pi
                indices = torch.argmin(torch.abs(diff), dim=-1)
                expanded_levels = levels.expand(*wrapped_phase.shape, levels.shape[-1])
                return torch.gather(expanded_levels, dim=-1, index=indices.unsqueeze(-1)).squeeze(-1)

        @staticmethod
        def torchNearestPhaseSTE(phase, quant_levels):
                nearest_phase = CGHGenerator.torchNearestPhase(phase, quant_levels)
                return nearest_phase.detach() + phase - phase.detach()

        @staticmethod
        def score_phase(phase, lut, s=5.0, func='sigmoid'):
                wrapped_phase = torch.remainder(phase + np.pi, 2 * np.pi) - np.pi
                lut_values = lut.to(device=phase.device, dtype=phase.dtype)
                while(lut_values.dim() < wrapped_phase.dim() + 1):
                        lut_values = lut_values.unsqueeze(0)
                diff = wrapped_phase.unsqueeze(-1) - 2*np.pi*lut_values
                diff = ((diff + np.pi) % (2 * np.pi)) - np.pi 
                diff /= np.pi  # Normalize

                if func == 'sigmoid':
                        z = s * diff
                        scores = torch.sigmoid(z) * (1 - torch.sigmoid(z)) * 4
                elif func == 'log':
                        scores = -torch.log(diff.abs() + 1e-20) * s
                elif func == 'poly':
                        scores = (1 - diff.abs()) ** s
                elif func == 'sine':
                        scores = torch.cos(np.pi * (s * diff).clamp(-1., 1.))
                elif func == 'chirp':
                        scores = 1 - torch.cos(np.pi * (1 - diff.abs()) ** s)
                else:
                        raise ValueError(f"Unknown scoring function: {func}")

                return scores * 300 * s

        # Gumbelsoftmax Quantization
        class GumbelSoftmaxQuantization(nn.Module):
                def __init__(self, lut = None, num_levels=16, tau=1.0, hard=True, score_func='sigmoid', s=5.0):
                        super(CGHGenerator.GumbelSoftmaxQuantization, self).__init__()
                        
                        # Number of quantization levels
                        self.num_levels = num_levels

                        # Temperature Variable
                        self.tau = tau
                        self.tauInit = tau
                        # Class configuration
                        self.hard = hard
                        self.score_func = score_func
                        self.s = s
                        if(lut is not None):
                                self.quant_levels = lut
                        else:
                                self.quant_levels = nn.Parameter(torch.linspace(0, 2* torch.pi, num_levels), requires_grad=False)

                def anneal_temperature(self, annealRate, tauInitial, tauMin):
                        self.tau = tauInitial * torch.exp(-torch.log(tauInitial/tauMin) * annealRate)
                
                def forward(self, phase):
                        # Compute logits using the scoring function
                        scores = CGHGenerator.score_phase(phase, self.quant_levels, s=self.tauInit/self.tau, func=self.score_func)
                        quant_levels = self.quant_levels
                        while(quant_levels.dim() < scores.dim()):
                                quant_levels = quant_levels.unsqueeze(0)

                        # Apply Gumbel noise for softmax sampling
                        gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-20) + 1e-20)

                        # Handle inf problem for high scores
                        up_scores = (scores + gumbel_noise)/self.tau
                        up_scores = up_scores - torch.max(up_scores, dim=-1, keepdim=True)[0]

                        soft_assignments = torch.nn.functional.softmax(up_scores, dim=-1)

                        if self.hard:
                                # Convert soft assignments to hard one-hot encoding
                                indices = torch.argmax(soft_assignments, dim=-1, keepdim=True)
                                hard_assignments = torch.zeros_like(soft_assignments).scatter_(-1, indices, 1.0)
                                probs = hard_assignments.detach() + soft_assignments - soft_assignments.detach()
                        else:
                                probs = soft_assignments

                        # Compute quantized phase
                        quantized_phase = torch.sum(2*np.pi*probs * quant_levels, dim=-1)

                        return quantized_phase

                def deterministic(self, phase):
                        scores = CGHGenerator.score_phase(phase, self.quant_levels, s=self.tauInit/self.tau, func=self.score_func)
                        quant_levels = self.quant_levels
                        while(quant_levels.dim() < scores.dim()):
                                quant_levels = quant_levels.unsqueeze(0)

                        indices = torch.argmax(scores, dim=-1, keepdim=True)
                        hard_assignments = torch.zeros_like(scores).scatter_(-1, indices, 1.0)
                        return torch.sum(2*np.pi*hard_assignments * quant_levels, dim=-1)

        # Quantize phase using a defined method. Returns the object to the class.
        def torchquantize(self, lut=None, levels=None, quantizeMethod = "GUMBELSOFTMAX", tau=1.0):
                CGHGenerator.requireTorch("Torch-based quantization")
                
                # Choose method of quantization
                if(quantizeMethod.upper() == "GUMBELSOFTMAX"):
                        quantize_method = CGHGenerator.GumbelSoftmaxQuantization(lut=lut, num_levels=levels, tau=tau, hard=True).to(self.MLDevice)
                
                return quantize_method
        
        def getReconstructionTarget(self, shape):
                if(hasattr(self, "imTarget_recon") and self.imTarget_recon is not None and self.imTarget_recon.shape == shape):
                        return self.imTarget_recon
                return self.imTarget

        # Reconstruct hologram from phase
        def reconstruct(self, phase, propMethod = "FOURIER"):

                E_hp = np.ones(phase.shape)*np.exp(1j*phase)

                if(propMethod.upper() == "FOURIER"):
                        E_ip = self.prop("forward", E_hp, "FOURIER")
                elif(propMethod.upper() == "ASM"):
                        E_ip = self.prop("forward", E_hp, "ASM")
                elif(propMethod.upper() == "COSINE"):
                        E_ip = dct(dct(E_hp.T, norm='ortho').T, norm='ortho')

                I = CGHGenerator.fieldToIntensity(E_ip)
                target_ref = self.getReconstructionTarget(I.shape)
                _, I = CGHGenerator.scaleIntensityToTarget(I, target_ref)

                return I,E_ip

        def createIllumination(self, illumType = "UNIFORM", ill_x0 = 0, ill_y0 = 0, ill_sigma_x = 1, ill_sigma_y = 1,
                               illuminationProfile = None, randomStd = 0.25, normalizeMean = False, eps = 1e-6):
                self.targetIllumination = self._buildIlluminationProfile(
                        illumType=illumType,
                        ill_x0=ill_x0,
                        ill_y0=ill_y0,
                        ill_sigma_x=ill_sigma_x,
                        ill_sigma_y=ill_sigma_y,
                        illuminationProfile=illuminationProfile,
                        randomStd=randomStd,
                        normalizeMean=normalizeMean,
                        eps=eps
                )
                self.optimizedIllumination = self.targetIllumination.copy()
                self.illuminationWasOptimized = False
                return self.targetIllumination.copy()

#########################
## Wavefront Functions ##
#########################

        @staticmethod
        def createSteerPhase(pitchW, pitchH, w, h, theta_x, theta_y, lambdaLoc):
                x = range(int(-w/2), int(w/2))
                y = range(int(-h/2), int(h/2))
                X, Y = np.meshgrid(x, y)

                u = X * pitchW
                v = -Y * pitchH
                
                z_rotx = -u*np.tan(np.radians(theta_x))
                z_roty = v*np.tan(np.radians(theta_y))
                tilt_height = z_rotx + z_roty
                tilt_height = tilt_height - np.max(tilt_height[:])

                idx_p = tilt_height >= 0
                idx_n = tilt_height < 0

                tilt_phase = np.zeros(tilt_height.shape)
                tilt_phase[idx_p] = -np.mod(tilt_height[idx_p], lambdaLoc/2) * (2*np.pi) / (lambdaLoc/2)
                tilt_phase[idx_n] = np.mod(-tilt_height[idx_n], lambdaLoc/2) * (2*np.pi) / (lambdaLoc/2)
                tilt_phase = np.mod(tilt_phase - np.min(tilt_phase[:]), 2*np.pi)

                return tilt_phase, tilt_height

######################
## Helper Functions ##
######################

        @staticmethod
        def quantizeCircularPhase(sig, state_levels):
                sig = np.asarray(sig, dtype=np.float64).reshape(-1)
                state_levels = np.mod(np.asarray(state_levels, dtype=np.float64).reshape(-1), 1.0)

                if(state_levels.size == 0):
                        raise ValueError("state_levels must contain at least one valid phase level.")

                if(state_levels.size > 1 and np.all(np.diff(state_levels) > 0)):
                        boundaries = 0.5 * (state_levels[:-1] + state_levels[1:])
                        sig_mod = np.mod(sig, 1.0)
                        indices = np.searchsorted(boundaries, sig_mod, side="left")
                        wrap_boundary = np.mod(0.5 * (state_levels[-1] + state_levels[0] + 1.0), 1.0)

                        if(wrap_boundary > state_levels[-1]):
                                indices[sig_mod >= wrap_boundary] = 0
                        elif(wrap_boundary < state_levels[0]):
                                indices[sig_mod < wrap_boundary] = 0
                        else:
                                diff = np.remainder(sig_mod[:, None] - state_levels[None, :] + 0.5, 1.0) - 0.5
                                indices = np.argmin(np.abs(diff), axis=1)

                        indices = indices.astype(np.float64)
                        quants = state_levels[indices.astype(np.int64)]
                        return indices, quants

                diff = np.remainder(sig[:, None] - state_levels[None, :] + 0.5, 1.0) - 0.5
                indices = np.argmin(np.abs(diff), axis=1).astype(np.float64)
                quants = state_levels[indices.astype(np.int64)]

                return indices, quants

        # Discretize phase:
        def discretePhase(self, phase, nLevel, pLevel):
                phaseq = np.zeros(np.shape(phase), dtype=np.float64)
                stateq = np.zeros(np.shape(phase), dtype=np.float64)

                phase = np.mod(phase, 2*np.pi)/(2*np.pi)
                phase_level = np.asarray(pLevel, dtype=np.float64)
                self.CGH_output_disc = []

                if(phase_level.ndim > 1):
                        num_luts = phase_level.shape[0]
                        for i in range(0, np.size(phase, 1)):
                                if(num_luts == 1):
                                        lut_idx = 0
                                else:
                                        lut_idx = min(i % num_luts, num_luts - 1)

                                state_levels = phase_level[lut_idx, :nLevel]
                                stateq[:, i], phaseq[:, i] = CGHGenerator.quantizeCircularPhase(phase[:, i], state_levels)
                else:
                        state_levels = phase_level[:nLevel]
                        stateq_flat, phaseq_flat = CGHGenerator.quantizeCircularPhase(phase.reshape(-1), state_levels)
                        stateq = stateq_flat.reshape(phase.shape)
                        phaseq = phaseq_flat.reshape(phase.shape)

                phaseq = np.mod(phaseq, 1.0)
                stateq = np.mod(stateq, nLevel)
                self.CGH_output_disc = stateq
                return phaseq, stateq
        
        # Writes the CGH to a desired filename
        def writeCGHToFile(self, filename, binary = False):
                # Make sure CGH is 8-bit
                outputCGH = cv2.normalize(self.CGH_mapped, None, 0, 255, cv2.NORM_MINMAX)
                outputCGH = outputCGH.astype(np.uint8)

                # Threshold the image to make binary
                _, binaryCGH = cv2.threshold(outputCGH, 127, 255, cv2.THRESH_BINARY)
                binaryCGH = binaryCGH.astype(np.uint8)

                if(self.CGH_is_sequence and len(outputCGH.shape) == 3 and outputCGH.shape[2] > 1):
                        frames = []
                        for idx in range(outputCGH.shape[2]):
                                if(binary):
                                        frames.append(binaryCGH[:, :, idx].astype(np.uint8))
                                else:
                                        frames.append(outputCGH[:, :, idx].astype(np.uint8))

                        ext = os.path.splitext(filename)[1].lower()
                        if(ext in [".tif", ".tiff"]):
                                pil_frames = [Image.fromarray(frame) for frame in frames]
                                if(binary):
                                        pil_frames = [frame.convert('1') for frame in pil_frames]
                                pil_frames[0].save(filename, save_all=True, append_images=pil_frames[1:], compression="tiff_lzw")
                        else:
                                root, ext = os.path.splitext(filename)
                                if(ext == ""):
                                        ext = ".bmp"
                                for idx, frame in enumerate(frames):
                                        out_fn = root + "_frame" + str(idx).zfill(3) + ext
                                        if(binary):
                                                Image.fromarray(frame).convert('1').save(out_fn)
                                        else:
                                                cv2.imwrite(out_fn, frame)
                        return

                # Write using OpenCV to maintain legacy functionality
                if(not binary):
                        ext = os.path.splitext(filename)[1].lower()
                        if(ext in [".tif", ".tiff"]):
                                cv2.imwrite(filename, outputCGH, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
                        else:
                                cv2.imwrite(filename, outputCGH)
                # Write using PIL for single bit plane
                else:
                        # Use PIL to write binary image
                        pilImage = Image.fromarray(binaryCGH)
                        pilImage = pilImage.convert('1')
                        pilImage.save(filename, 'BMP')

        # Write a variable to a file
        def writeVarToFile(self, filename, Var):
                # Make sure CGH is 8-bit
                outputCGH = cv2.normalize(Var, None, 0, 255, cv2.NORM_MINMAX)
                outputCGH = outputCGH.astype(np.uint8)
                # Write to file
                cv2.imwrite(filename, outputCGH)

        # quantiz function
        @staticmethod
        def quantiz(sig, partition, codebook):
                index = []
                quants = []
                for s in sig:
                        idx = 0
                        while idx < len(partition) and s > partition[idx]:
                                idx += 1
                        index.append(idx)
                        quants.append(codebook[idx])
                return np.array(index), np.array(quants)

        # Cartesian to polar coordinates
        @staticmethod
        def cart2pol(x, y):
                rho = np.sqrt(x**2 + y**2)
                phi = np.arctan2(y, x)
                return(rho, phi)

        # Polar to cartesian coordinates
        @staticmethod
        def pol2cart(rho, phi):
                x = rho * np.cos(phi)
                y = rho * np.sin(phi)
                return(x, y)

        # Mean and std deviation matching
        @staticmethod
        def MeanAndStdMatch(Im_in, Im_target):
                target_mean = np.mean(Im_target)
                baseline_mean = np.mean(Im_in)
                target_std = np.std(Im_target)
                baseline_std = np.std(Im_in)

                Im_out = (Im_in - baseline_mean) * (target_std / baseline_std) + target_mean

                return Im_out

        # Normalize a vector/matrix
        @staticmethod
        def normalize(x):
                return (x - np.min(x)) / (np.max(x) - np.min(x))
