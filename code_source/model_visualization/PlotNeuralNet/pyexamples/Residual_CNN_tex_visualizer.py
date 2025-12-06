import sys
sys.path.append('../')
from pycore.tikzeng import *

# Residual CNN Architecture
# Input: (6, 10, 1)
# Branch 1 (Conv):
#   Conv1: (6, 8, 2) -> Pool -> (6, 4, 2)
#   Conv2: (6, 2, 4) -> Pool -> (6, 1, 4)
#   Flatten: 24
# Branch 2 (Skip):
#   Slice -> (6, 1, 1)
#   Flatten: 6
# Concatenate: 30
# Dense: 12
# Output: 5

arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    
    # Input
    to_Conv("input", 10, 6, offset="(0,0,0)", to="(0,0,0)", height=6, depth=10, width=1, caption="Input" ),
    
    # --- Branch 1 (Conv) ---
    # Shifted up slightly
    to_Conv("conv1", 8, 6, offset="(2,2,0)", to="(input-east)", height=6, depth=8, width=2, caption="Conv1" ),
    to_connection("input", "conv1"),
    
    to_Pool("pool1", offset="(1,0,0)", to="(conv1-east)", height=6, depth=4, width=2, caption="Pool1" ),
    to_connection("conv1", "pool1"),
    
    to_Conv("conv2", 2, 6, offset="(1,0,0)", to="(pool1-east)", height=6, depth=2, width=4, caption="Conv2" ),
    to_connection("pool1", "conv2"),
    
    to_Pool("pool2", offset="(1,0,0)", to="(conv2-east)", height=6, depth=1, width=4, caption="Pool2" ),
    to_connection("conv2", "pool2"),
    
    to_Conv("conv_flat", 24, 1, offset="(1,0,0)", to="(pool2-east)", height=1, depth=12, width=1, caption="Flat (24)" ),
    to_connection("pool2", "conv_flat"),
    
    # --- Branch 2 (Skip) ---
    # Shifted down
    to_Conv("skip", 1, 6, offset="(4,-4,0)", to="(input-east)", height=6, depth=1, width=1, caption="Skip (Last Frame)" ),
    to_connection("input", "skip"),
    
    to_Conv("skip_flat", 6, 1, offset="(1,0,0)", to="(skip-east)", height=1, depth=3, width=1, caption="Flat (6)" ),
    to_connection("skip", "skip_flat"),
    
    # --- Concatenate ---
    # Positioned relative to conv_flat but shifted down to meet in the middle
    to_Conv("concat", 30, 1, offset="(2,-2,0)", to="(conv_flat-east)", height=1, depth=15, width=2, caption="Concat (30)" ),
    
    # Connections
    to_connection("conv_flat", "concat"),
    to_connection("skip_flat", "concat"),
    
    # Dense
    to_Conv("dense", 12, 1, offset="(2,0,0)", to="(concat-east)", height=1, depth=6, width=2, caption="Dense (12)" ),
    to_connection("concat", "dense"),
    
    # Output
    to_SoftMax("output", 5 ,"(2,0,0)", "(dense-east)", caption="Output (5)", height=1, depth=3, width=2 ),
    to_connection("dense", "output"),
    
    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
