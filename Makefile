PKG := src/cimple_rtl
RTL := $(PKG)/rtl

.PHONY: luts test lint clean

luts:
	uv run python -m cimple_rtl.model.gen_luts --score-scale 0.05

test: luts
	uv run $(MAKE) -C $(PKG)/tb regress

lint:
	verilator --lint-only -Wall -Wno-DECLFILENAME -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM \
	  --top-module cim_core \
	  $(RTL)/cimple_pkg.sv $(RTL)/cim_macro.sv $(RTL)/cim_partition.sv $(RTL)/cim_core.sv
	verilator --lint-only -Wall -Wno-DECLFILENAME -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM \
	  --top-module split_softmax \
	  $(RTL)/cimple_pkg.sv $(RTL)/quant_unit.sv $(RTL)/lut_rom.sv $(RTL)/split_softmax.sv

clean:
	rm -rf $(PKG)/tb/sim_build build $(PKG)/tb/results.xml
	find $(PKG) -name __pycache__ -prune -exec rm -rf {} +
