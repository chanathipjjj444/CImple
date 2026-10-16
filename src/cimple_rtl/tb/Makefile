# cocotb + Verilator regression for the CIMple re-implementation.
#
#   make                 # run every test with Verilator
#   make SIM=icarus      # same tests on Icarus (slower, weaker SV support)
#   make MODULE=test_cim_partition
#   make waves           # re-run with VCD dumping on
#
TOPDIR  := $(shell cd .. && pwd)
SIM     ?= verilator
TOPLEVEL_LANG ?= verilog

VERILOG_SOURCES := \
  $(TOPDIR)/rtl/cimple_pkg.sv \
  $(TOPDIR)/rtl/bitcell.sv \
  $(TOPDIR)/rtl/oai221.sv \
  $(TOPDIR)/rtl/cim_macro.sv \
  $(TOPDIR)/rtl/cim_partition.sv \
  $(TOPDIR)/rtl/cim_core.sv \
  $(TOPDIR)/rtl/quant_unit.sv \
  $(TOPDIR)/rtl/lut_rom.sv \
  $(TOPDIR)/rtl/split_softmax.sv

MODULE   ?= test_cim_partition
TOPLEVEL ?= cim_partition

# One build dir per top level, otherwise switching TOPLEVEL silently reuses
# the previous elaboration and cocotb fails with "Can not find root handle".
SIM_BUILD := sim_build/$(SIM)_$(TOPLEVEL)

# Tracing costs a lot of compile time; opt in with `make WAVES=1`.
WAVES ?= 0
ifeq ($(WAVES),1)
  EXTRA_ARGS += --trace --trace-structs
endif
EXTRA_ARGS   += -j $(shell nproc)
COMPILE_ARGS += -Wno-fatal --timing
# LUT contents are elaboration-time parameters.
ifeq ($(TOPLEVEL),split_softmax)
  COMPILE_ARGS += -GEXP_HEX='"$(LUT_DIR)/exp_lut.hex"'
  COMPILE_ARGS += -GRECIP_HEX='"$(LUT_DIR)/recip_lut.hex"'
endif

export PYTHONPATH := $(TOPDIR)/..:$(PYTHONPATH)
export LUT_DIR    := $(TOPDIR)/../../build

include $(shell cocotb-config --makefiles)/Makefile.sim

.PHONY: regress
# `make regress` runs the whole suite. Do not call this target `all` --
# cocotb's Makefile.sim already owns that name.
regress:
	$(MAKE) MODULE=test_cim_partition TOPLEVEL=cim_partition
	$(MAKE) MODULE=test_quant_unit    TOPLEVEL=quant_unit
	$(MAKE) MODULE=test_split_softmax TOPLEVEL=split_softmax
	$(MAKE) MODULE=test_cim_core      TOPLEVEL=cim_core
