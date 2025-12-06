import sys
sys.path.append('../')
from pycore.tikzeng import *

# CNN Architecture
# Input: (6, 20, 1)
# Conv1: (6, 18, 2) -> Pool -> (6, 9, 2)
# Conv2: (6, 7, 4) -> Pool -> (6, 3, 4)
# Flatten: 6*3*4 = 72
# Dense: 16
# Output: 5

arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    
    # Input
    to_Conv("input", 20, 6, offset="(0,0,0)", to="(0,0,0)", height=6, depth=20, width=1, caption="Input" ),
    
    # Conv 1
    to_Conv("conv1", 18, 6, offset="(2,0,0)", to="(input-east)", height=6, depth=18, width=2, caption="Conv1" ),
    to_connection("input", "conv1"),
    
    # Pool 1
    to_Pool("pool1", offset="(1,0,0)", to="(conv1-east)", height=6, depth=9, width=2, caption="Pool1" ),
    to_connection("conv1", "pool1"),
    
    # Conv 2
    to_Conv("conv2", 7, 6, offset="(2,0,0)", to="(pool1-east)", height=6, depth=7, width=4, caption="Conv2 (4 filters)" ),
    to_connection("pool1", "conv2"),
    
    # Pool 2
    to_Pool("pool2", offset="(1,0,0)", to="(conv2-east)", height=6, depth=3, width=4, caption="Pool2" ),
    to_connection("conv2", "pool2"),
    
    # Flatten
    to_Conv("flatten", 72, 1, offset="(2,0,0)", to="(pool2-east)", height=1, depth=25, width=1, caption="Flatten (72)" ),
    to_connection("pool2", "flatten"),
    
    # Dense
    to_Conv("dense", 16, 1, offset="(2,0,0)", to="(flatten-east)", height=1, depth=10, width=2, caption="Dense (16)" ),
    to_connection("flatten", "dense"),
    
    # Output
    to_SoftMax("output", 5 ,"(2,0,0)", "(dense-east)", caption="Output (5)", height=1, depth=5, width=2 ),
    to_connection("dense", "output"),
    
    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
