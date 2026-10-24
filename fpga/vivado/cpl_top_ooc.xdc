# ---------------------------------------------------------------------------
# cpl_top_ooc.xdc -- out-of-context timing constraints for cpl_top
#
# Used only by synth_ooc.tcl (read with -mode out_of_context). In the full
# CIPS block design the clock comes from CIPS pl0_ref_clk and IP integrator
# constrains it automatically, so this file is NOT used there.
# ---------------------------------------------------------------------------

# 200 MHz single clock domain (ARCH.md §2, [A])
create_clock -period 5.000 -name clk_i [get_ports clk_i]

# [A] Port budgets: in the block design every cpl_top port talks to a
# registered SmartConnect / proc_sys_reset output (or input), so give the
# outside world 1 ns of the 5 ns cycle on each side. Without these the AXI
# port paths would be unconstrained and invisible in the timing report.
set cpl_in_ports [get_ports -filter {DIRECTION == IN && NAME != clk_i}]
set_input_delay  -clock clk_i 1.000 $cpl_in_ports
set_output_delay -clock clk_i 1.000 [get_ports -filter {DIRECTION == OUT}]
