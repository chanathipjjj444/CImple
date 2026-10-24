# ---------------------------------------------------------------------------
# synth_ooc.tcl -- out-of-context synthesis of cpl_top for the Versal part
#
# FIRST thing to run on a machine with Vivado: it needs no board, no CIPS and
# no block design, and it tells you whether the RTL fits and closes 200 MHz.
#
#   cd fpga/vivado
#   vivado -mode batch -source synth_ooc.tcl -notrace -log out/synth_ooc.log -journal out/synth_ooc.jou
#   vivado -mode batch -source synth_ooc.tcl -tclargs -part xcvp1202-vsva2785-2MP-e-S -jobs 16
#   vivado -mode batch -source synth_ooc.tcl -tclargs -impl 1          ;# + OOC place & route
#   vivado -mode batch -source synth_ooc.tcl -tclargs -generic {GBUF_BYTES=65536 PIPE_TREE=1}
#
# Options (-tclargs) / environment:
#   -part    / CPL_PART   default xcvp1202-vsva2785-2MP-e-S (see common.tcl)
#   -out     / CPL_OUT    report directory, default fpga/vivado/out
#   -jobs    / CPL_JOBS   threads (Vivado caps at 8 for most steps), default 8
#   -impl    / CPL_IMPL   1 = also opt/place/phys_opt/route out of context
#   -generic / CPL_GENERIC  space-separated NAME=VALUE list of cpl_top
#                         parameters (names: see fpga/rtl/cpl_top.sv)
#
# Outputs in <out>/:
#   ooc_synth.dcp          checkpoint
#   ooc_util.rpt           flat utilization (LUT/FF/DSP/BRAM/URAM sections)
#   ooc_util_hier.rpt      hierarchical utilization (per cpl_* instance)
#   ooc_ram_util.rpt       RAM inference detail (if the command exists)
#   ooc_primitives.rpt     quick DSP / RAMB / URAM / LUTRAM / FF / latch counts
#   ooc_timing_summary.rpt timing summary at 5.000 ns
#   ooc_timing_paths.rpt   50 worst setup paths
#   ooc_summary.txt        one-screen PASS/FAIL summary (also printed)
#   ooc_impl_*             the same after route when -impl 1
#
# STATUS: written against Vivado 2024.2 / 2025.1 documentation; NOT run
# (no Vivado on the authoring machine). Syntax was checked under tclsh with
# stub Vivado commands only.
# ---------------------------------------------------------------------------

set here [file dirname [file normalize [info script]]]
source [file join $here common.tcl]
source [file join $here files.tcl]

cpl_check_args $argv {part out jobs impl generic}
set part     [cpl_getopt $argv part    $CPL_DEFAULT_PART CPL_PART]
set out      [file normalize [cpl_getopt $argv out [file join $here out] CPL_OUT]]
set jobs     [cpl_getopt $argv jobs    8 CPL_JOBS]
set do_impl  [cpl_getopt $argv impl    0 CPL_IMPL]
set generics [cpl_getopt $argv generic "" CPL_GENERIC]
set top      cpl_top
set period   5.000

cpl_vivado_banner synth_ooc.tcl
puts "cpl: part=$part out=$out jobs=$jobs impl=$do_impl generics={$generics}"

cpl_require_part $part
cpl_check_rtl_files
file mkdir $out

set_param general.maxThreads [expr {$jobs > 8 ? 8 : $jobs}]

# ---- read sources (cpl_pkg first) ------------------------------------------
read_verilog -sv $CPL_RTL_FILES
read_xdc -mode out_of_context [file join $here cpl_top_ooc.xdc]

set gargs {}
foreach g $generics { lappend gargs -generic $g }

# ---- synthesis --------------------------------------------------------------
# -mode out_of_context: no I/O buffers, ports stay internal nets (as they are
# in the block design). flatten_hierarchy rebuilt (the default) keeps the
# hierarchical utilization report meaningful.
synth_design -top $top -part $part -mode out_of_context \
    -flatten_hierarchy rebuilt {*}$gargs

write_checkpoint -force [file join $out ooc_synth.dcp]

# ---- reports ----------------------------------------------------------------
proc cpl_reports {out tag period} {
    report_utilization -file [file join $out ${tag}_util.rpt]
    report_utilization -hierarchical -hierarchical_depth 6 \
        -file [file join $out ${tag}_util_hier.rpt]
    # report_ram_utilization exists in recent Vivado; do not fail without it
    if {[catch {report_ram_utilization -file [file join $out ${tag}_ram_util.rpt]} e]} {
        puts "cpl: note: report_ram_utilization unavailable ($e)"
    }
    report_timing_summary -max_paths 20 -report_unconstrained \
        -file [file join $out ${tag}_timing_summary.rpt]
    report_timing -setup -max_paths 50 -nworst 1 -sort_by group \
        -file [file join $out ${tag}_timing_paths.rpt]
    report_clocks -file [file join $out ${tag}_clocks.rpt]

    # Quick primitive counts. REF_NAME patterns, so they survive the
    # different Versal primitive names (DSP58/DSPCPLX/DSPFP32, RAMB18E5/
    # RAMB36E5, URAM288E5, RAM32M/RAM64X1D...). ooc_util.rpt is authoritative.
    set fh [open [file join $out ${tag}_primitives.rpt] w]
    set n_latch 0
    foreach {label filt} {
        "LUTs"                {IS_PRIMITIVE && REF_NAME =~ LUT*}
        "flip-flops (FD*)"    {IS_PRIMITIVE && REF_NAME =~ FD*}
        "LATCHES (must be 0)" {IS_PRIMITIVE && REF_NAME =~ LD*}
        "DSP (DSP*)"          {IS_PRIMITIVE && REF_NAME =~ DSP*}
        "block RAM (RAMB*)"   {IS_PRIMITIVE && REF_NAME =~ RAMB*}
        "UltraRAM (URAM*)"    {IS_PRIMITIVE && REF_NAME =~ URAM*}
        "LUTRAM (RAM32*/RAM64*/RAM128*/RAM256*)" {IS_PRIMITIVE && (REF_NAME =~ RAM32* || REF_NAME =~ RAM64* || REF_NAME =~ RAM128* || REF_NAME =~ RAM256*)}
        "SRL (SRL*)"          {IS_PRIMITIVE && REF_NAME =~ SRL*}
    } {
        set n [llength [get_cells -hierarchical -quiet -filter $filt]]
        cpl_tee $fh [format "%-42s %8d" $label $n]
        if {[string match LATCHES* $label]} { set n_latch $n }
    }
    close $fh

    # Worst setup / hold slack
    set wns ""
    set whs ""
    set sp [get_timing_paths -quiet -setup -max_paths 1 -nworst 1]
    set hp [get_timing_paths -quiet -hold  -max_paths 1 -nworst 1]
    if {[llength $sp] > 0} { set wns [get_property SLACK [lindex $sp 0]] }
    if {[llength $hp] > 0} { set whs [get_property SLACK [lindex $hp 0]] }

    set fh [open [file join $out ${tag}_summary.txt] w]
    cpl_tee $fh "cpl: ==== $tag summary (clk_i period $period ns) ===="
    if {$wns eq ""} {
        cpl_tee $fh "cpl: WNS: no constrained setup path found -- check that\
                     clk_i exists in cpl_top and cpl_top_ooc.xdc was applied"
    } elseif {$wns < 0} {
        cpl_tee $fh "cpl: TIMING FAIL: WNS = $wns ns (worst paths: ${tag}_timing_paths.rpt)"
    } else {
        cpl_tee $fh "cpl: TIMING PASS: WNS = $wns ns"
    }
    if {$whs ne ""} { cpl_tee $fh "cpl: WHS = $whs ns" }
    if {$n_latch > 0} {
        cpl_tee $fh "cpl: ERROR: $n_latch latch(es) inferred -- see the synthesis log (Synth 8-327)"
    }
    close $fh
}

cpl_reports $out ooc $period
puts "cpl: note: post-synthesis timing has no real routing; run with\
      -impl 1 (or the full build_versal.tcl flow) before trusting WNS."

# ---- optional OOC implementation --------------------------------------------
# [A] Without HD.CLK_SRC on clk_i, OOC place/route treats the clock as ideally
# routed from an unspecified buffer: fine for a datapath timing estimate,
# not a sign-off number (the build_versal.tcl flow is).
if {$do_impl} {
    opt_design
    place_design
    phys_opt_design
    route_design
    write_checkpoint -force [file join $out ooc_impl_routed.dcp]
    cpl_reports $out ooc_impl $period
}

puts "cpl: synth_ooc.tcl done, reports in $out"
