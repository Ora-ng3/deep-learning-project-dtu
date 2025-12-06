import sys
sys.path.append('../')
from pycore.tikzeng import *

# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    
    # Input Layer (Flattened input 10x6 = 60)
    # Represented as a long vector
    to_Conv("input", 60, "", offset="(0,0,0)", to="(0,0,0)", height=4, depth=15, width=2, caption="Input (60)" ),
    
    # Dense 1 (32 units)
    to_Conv("dense1", 32, "", offset="(2,0,0)", to="(input-east)", height=16, depth=2, width=2, caption="Dense 32" ),
    to_connection( "input", "dense1"),
    
    # Dense 2 (16 units)
    to_Conv("dense2", 16, "", offset="(2,0,0)", to="(dense1-east)", height=8, depth=2, width=2, caption="Dense 16" ),
    to_connection( "dense1", "dense2"),
    
    # Output (5 units) - Using SoftMax style for the final layer
    to_SoftMax("output", 5 ,"(2,0,0)", "(dense2-east)", caption="Output (5)", height=4, depth=2, width=2 ),
    to_connection("dense2", "output"),
    
    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
