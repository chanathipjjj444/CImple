# cocotb + Verilator regression for fpga/rtl (the cpl_* VP1202 design).
#
#   cd <repo> && unset VIRTUAL_ENV
#   uv run make -C fpga/tb MODULE=test_cpl_core TOPLEVEL=cpl_core
#   uv run make -C fpga/tb regress
#   uv run make -C fpga/tb WAVES=1 MODULE=... TOPLEVEL=...
#
# Each top level compiles only the files it needs (SRCS_<toplevel>), so a
# half-written module elsewhere in rtl/ never breaks an unrelated test.

RTL     := $(abspath ../rtl)
HOST    := $(abspath ../host)
REPO    := $(abspath ../..)
SIM     ?= verilator
TOPLEVEL_LANG ?= verilog

SRCS_cpl_partition := cpl_pkg.sv cpl_bitcell.sv cpl_oai221.sv cpl_macro.sv cpl_partition.sv
SRCS_cpl_core      := $(SRCS_cpl_partition) cpl_core.sv
SRCS_cpl_quant     := cpl_pkg.sv cpl_quant.sv
SRCS_cpl_lut       := cpl_pkg.sv cpl_lut.sv
SRCS_cpl_post      := cpl_pkg.sv cpl_quant.sv cpl_lut.sv cpl_post.sv
SRCS_cpl_top       := $(SRCS_cpl_core) cpl_quant.sv cpl_lut.sv cpl_post.sv \
                      cpl_gbuf.sv cpl_imem.sv cpl_engine.sv cpl_axil.sv cpl_top.sv

MODULE   ?= test_cpl_core
TOPLEVEL ?= cpl_core

VERILOG_SOURCES := $(addprefix $(RTL)/,$(SRCS_$(TOPLEVEL)))

# one build directory per (TOPLEVEL, PARAMS): changing PARAMS alone used to
# reuse a stale build. SIM_BUILD=... on the command line still overrides;
# SIM_BUILD_ROOT=... moves the whole tree (use it for `regress`, whose steps
# each need their own directory, e.g. when several users share the checkout).
# cocotb's Makefile.inc EXPORTS SIM_BUILD and COCOTB_RESULTS_FILE, so every
# `$(MAKE)` step of `regress` used to inherit the parent's directory through
# the environment (a plain ?= then kept it) and ran another top's Vtop. Both
# are therefore recomputed unless they were given on the command line.
empty :=
space := $(empty) $(empty)
PARAMS_TAG := $(subst $(space),_,$(strip $(subst -G,,$(subst =,,$(subst ",,$(subst \",,$(PARAMS)))))))
SIM_BUILD_ROOT ?= sim_build
ifneq ($(origin SIM_BUILD),command line)
  override SIM_BUILD := $(SIM_BUILD_ROOT)/$(SIM)_$(TOPLEVEL)$(if $(PARAMS_TAG),_$(PARAMS_TAG))
endif
# per-build results file: concurrent runs must not read each other's results
ifneq ($(origin COCOTB_RESULTS_FILE),command line)
  override COCOTB_RESULTS_FILE := $(abspath $(SIM_BUILD))/results.xml
endif

WAVES ?= 0
ifeq ($(WAVES),1)
  EXTRA_ARGS += --trace --trace-structs
endif
EXTRA_ARGS   += -j $(shell nproc)
COMPILE_ARGS += -Wno-fatal --timing
# per-test parameter overrides, e.g. make PARAMS="-GPIPE_TREE=0"
COMPILE_ARGS += $(PARAMS)

export PYTHONPATH := $(HOST):$(REPO)/src:$(abspath .):$(PYTHONPATH)

include $(shell cocotb-config --makefiles)/Makefile.sim

# cpl_top: cocotb's --public-flat-rw over the 32k bitcells + GBUF makes the
# Verilator build and the simulation slow, and the top-level tests only use
# the AXI4-Lite ports. Expose just cpl_top's own signals to VPI instead.
ifeq ($(TOPLEVEL),cpl_top)
  COMPILE_ARGS := $(filter-out --public-flat-rw,$(COMPILE_ARGS)) $(abspath cpl_top_public.vlt)
  CUSTOM_COMPILE_DEPS += $(abspath cpl_top_public.vlt)
endif

# the default-parameter (512 KiB GBUF) build runs a reduced set
TOP_DEFAULT_TESTS ?= 'test_regs|test_mem_readback|test_opcodes|test_encoder_head'

# `make regress REGRESS_GOAL=print-build` lists each step's build directory
# without building (check-regress-dirs asserts they are all different).
REGRESS_GOAL ?=
ULTRA := -GGBUF_RAM_STYLE=\"ultra\"

.PHONY: regress print-build check-regress-dirs synth-check
regress:
	$(MAKE) MODULE=test_cpl_partition TOPLEVEL=cpl_partition $(REGRESS_GOAL)
	$(MAKE) MODULE=test_cpl_partition TOPLEVEL=cpl_partition PARAMS="-GPIPE_TREE=1" $(REGRESS_GOAL)
	$(MAKE) MODULE=test_cpl_partition TOPLEVEL=cpl_partition PARAMS="-GPIPE_TREE=0" $(REGRESS_GOAL)
	$(MAKE) MODULE=test_cpl_core      TOPLEVEL=cpl_core $(REGRESS_GOAL)
	$(MAKE) MODULE=test_cpl_core      TOPLEVEL=cpl_core PARAMS="-GPIPE_TREE=0" $(REGRESS_GOAL)
	$(MAKE) MODULE=test_cpl_quant     TOPLEVEL=cpl_quant $(REGRESS_GOAL)
	$(MAKE) MODULE=test_cpl_quant     TOPLEVEL=cpl_quant PARAMS="-GLAT=5" $(REGRESS_GOAL)
	$(MAKE) MODULE=test_cpl_quant     TOPLEVEL=cpl_quant PARAMS="-GLAT=5 -GRND_FIX=1" $(REGRESS_GOAL)
	$(MAKE) MODULE=test_cpl_post      TOPLEVEL=cpl_post $(REGRESS_GOAL)
	$(MAKE) MODULE=test_cpl_lut       TOPLEVEL=cpl_lut $(REGRESS_GOAL)
	$(MAKE) MODULE=test_cpl_top       TOPLEVEL=cpl_top PARAMS="-GGBUF_BYTES=65536" $(REGRESS_GOAL)
	$(MAKE) MODULE=test_cpl_top       TOPLEVEL=cpl_top PARAMS="-GGBUF_BYTES=16384 -GGBUF_RD_LAT=1 -GGBUF_REQ_REG=0 -GPIPE_TREE=0" $(REGRESS_GOAL)
	$(MAKE) MODULE=test_cpl_top       TOPLEVEL=cpl_top PARAMS='-GGBUF_BYTES=65536 -GGBUF_RD_LAT=2 -GPIPE_TREE=1 $(ULTRA)' COCOTB_TEST_FILTER="$(TOP_DEFAULT_TESTS)" $(REGRESS_GOAL)
	$(MAKE) MODULE=test_cpl_top       TOPLEVEL=cpl_top COCOTB_TEST_FILTER="$(TOP_DEFAULT_TESTS)" $(REGRESS_GOAL)

print-build:
	@echo "BUILD $(TOPLEVEL) [$(PARAMS)] -> $(SIM_BUILD) results=$(COCOTB_RESULTS_FILE)"

# regression for the shared-SIM_BUILD bug: every regress step must get its own
# build directory and results file
check-regress-dirs:
	@$(MAKE) --no-print-directory -s regress REGRESS_GOAL=print-build > $(SIM_BUILD_ROOT)/.regress_dirs.txt
	@cat $(SIM_BUILD_ROOT)/.regress_dirs.txt
	@n=$$(grep -c '^BUILD ' $(SIM_BUILD_ROOT)/.regress_dirs.txt); \
	 u=$$(grep '^BUILD ' $(SIM_BUILD_ROOT)/.regress_dirs.txt | sed 's/.*-> \([^ ]*\) .*/\1/' | sort -u | wc -l); \
	 test $$n -ge 14 -a $$n -eq $$u && echo "check-regress-dirs: PASS ($$n steps, $$u build dirs)" \
	   || { echo "check-regress-dirs: FAIL ($$n steps, $$u distinct build dirs)"; exit 1; }

# synthesis-structure checks (yosys; no Vivado needed): URAM no-change
# mapping of cpl_gbuf, reset style of every module. See check_synth.py.
SV2V  ?= sv2v
YOSYS ?= yowasp-yosys
synth-check:
	SV2V="$(SV2V)" YOSYS="$(YOSYS)" python3 $(abspath check_synth.py) $(RTL) $(SIM_BUILD_ROOT)/synth_check
