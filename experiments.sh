#!/bin/bash

# Create directory reports if it does not exist
if [ ! -d "reports" ]; then
  mkdir reports
fi
# Delete everything in the reports directory
#rm -rf reports/*
#python3 tests.py
# For each python file in the generated-code directory
#python3 generated-code/T_1D.py generated-code/x_expsquare.npy generated-code/result.npy
#python3 generated-code/T_NC_1D.py generated-code/x_expsquare.npy generated-code/result.npy
#python3 generated-code/T_F_1D.py generated-code/x_expsquare.npy generated-code/result.npy
#python3 generated-code/T_B1_1D.py generated-code/x_expsquare.npy generated-code/result.npy
#python3 generated-code/T_B2_1D.py generated-code/x_expsquare.npy generated-code/result.npy
#python3 generated-code/T_B3_1D.py generated-code/x_expsquare.npy generated-code/result.npy
#python3 generated-code/H_1D.py generated-code/x_expsquare.npy generated-code/result.npy
#python3 generated-code/H_KPP1_1D.py generated-code/x_expsquare.npy generated-code/result.npy
#python3 generated-code/H_KPP2_1D.py generated-code/x_expsquare.npy generated-code/result.npy
#python3 generated-code/H_KPP3_1D.py generated-code/x_expsquare.npy generated-code/result.npy
python3 generated-code/W_2D.py generated-code/xy_expsquare.npy generated-code/xy_expsquare.npy generated-code/result.npy
python3 generated-code/W_B1_2D.py generated-code/xy_expsquare.npy generated-code/xy_expsquare.npy generated-code/result.npy
python3 generated-code/W_B2_2D.py generated-code/xy_expsquare.npy generated-code/xy_expsquare.npy generated-code/result.npy
