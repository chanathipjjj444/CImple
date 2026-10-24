# ---------------------------------------------------------------------------
# files.tcl -- RTL source list of the cpl_* design (fpga/rtl), in compile order
#
# Shared by synth_ooc.tcl and build_versal.tcl. Plain Tcl 8.5 (no Vivado
# commands), so it can be sourced from anywhere.
#
# Sets:
#   CPL_VIVADO_DIR   this directory (fpga/vivado), absolute
#   CPL_RTL_DIR      fpga/rtl, absolute
#   CPL_RTL_NAMES    module/file stems in compile order
#   CPL_RTL_FILES    absolute .sv paths in compile order (cpl_pkg first)
#   CPL_BD_SHIM      the Verilog-2001 port shim used as the IP-integrator
#                    module reference (see cpl_top_bd.v for why it exists)
# Defines:
#   cpl_check_rtl_files   errors out with a clear message if a file is missing
#
# Compile order: cpl_pkg must come first (every module imports it); the rest
# is leaf-to-root so that tools which do not sort (read_verilog in
# non-project mode elaborates after all files are read, so order only matters
# for the package) see definitions before use.
# ---------------------------------------------------------------------------

set CPL_VIVADO_DIR [file dirname [file normalize [info script]]]
set CPL_RTL_DIR    [file normalize [file join $CPL_VIVADO_DIR .. rtl]]

set CPL_RTL_NAMES {
    cpl_pkg
    cpl_bitcell
    cpl_oai221
    cpl_macro
    cpl_partition
    cpl_core
    cpl_quant
    cpl_lut
    cpl_post
    cpl_gbuf
    cpl_imem
    cpl_engine
    cpl_axil
    cpl_top
}

set CPL_RTL_FILES {}
foreach _n $CPL_RTL_NAMES {
    lappend CPL_RTL_FILES [file join $CPL_RTL_DIR ${_n}.sv]
}
unset _n

set CPL_BD_SHIM [file join $CPL_VIVADO_DIR cpl_top_bd.v]

proc cpl_check_rtl_files {{extra {}}} {
    global CPL_RTL_FILES
    set missing {}
    foreach f [concat $CPL_RTL_FILES $extra] {
        if {![file isfile $f]} { lappend missing $f }
    }
    if {[llength $missing] > 0} {
        error "cpl: missing RTL source file(s):\n  [join $missing "\n  "]\n\
               (fpga/rtl is incomplete -- run the cocotb regression\
               'make -C fpga/tb regress' first)"
    }
}
