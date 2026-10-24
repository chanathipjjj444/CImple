"""Host software for the CIMple-FPGA accelerator (fpga/docs/ARCH.md).

isa      -- instruction encode/decode, assembler, disassembler (ARCH.md §4)
sim      -- bit-exact ISA + AXI4-Lite register simulator (ARCH.md §3, §5)
compiler -- GBUF planner + matmul / encoder / decoder / cross-attention builders
driver   -- Device over Sim / /dev/mem-UIO / xsdb-JTAG MMIO backends
"""
from . import compiler, driver, isa, sim  # noqa: F401
from .driver import Device, DevMemBackend, SimBackend, XsdbBackend  # noqa: F401
from .isa import Acc, Err, Instr, Op, Out  # noqa: F401
from .sim import CplSim, SlvErr  # noqa: F401
