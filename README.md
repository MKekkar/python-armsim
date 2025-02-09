# python-armsim
making assembly run as slow as python

# details
this project implements a partial interpreter for a subset of armv6 assembly. 
see [test_sim.py](src/test_sim.py) for a very basic usage example.

# notes & warning
the interpreter is not faithful for certain instructions like misaligned loads or `pop {pc}`. 
no guarantees of correctness are made anywhere else and no maintenance is being performed on this code. 
use at your own risk.

very basic stack frames are supported but obviously there is no libc so things like `argc` and `argv` don't exist. 
registers and memory are randomized when used before initialization.
