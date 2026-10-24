# ---------------------------------------------------------------------------
# build_versal.tcl -- project-mode build: CIPS block design + cpl_top -> .pdi/.xsa
#
# Run synth_ooc.tcl first (it is faster and checks the part / licence / RTL).
#
#   cd fpga/vivado
#   vivado -mode batch -source build_versal.tcl -notrace -log out/build_versal.log -journal out/build_versal.jou
#   vivado -mode batch -source build_versal.tcl -tclargs -to bd      ;# stop after the block design
#   vivado -mode batch -source build_versal.tcl -tclargs -to synth   ;# stop after synthesis
#   vivado -mode batch -source build_versal.tcl -tclargs -pmc_ref_mhz 33.3333 -base 0xA4000000
#
# Options (-tclargs) / environment:
#   -part        / CPL_PART         default xcvp1202-vsva2785-2MP-e-S
#   -out         / CPL_OUT          output dir, default fpga/vivado/out
#   -jobs        / CPL_JOBS         parallel runs, default 8
#   -to          / CPL_TO           bd | synth | pdi (default pdi = everything)
#   -base        / CPL_BASE_ADDR    cpl window base, default 0xA4000000
#   -pmc_ref_mhz / CPL_PMC_REF_CLK_MHZ  PMC REF_CLK (BOARD-SPECIFIC), default 33.3333
#   -pl0_mhz     / CPL_PL0_MHZ      PL clock, default 200
#   -strategy    / CPL_IMPL_STRATEGY  impl_1 strategy, default "Vivado Implementation Defaults"
#
# Outputs in <out>/:
#   proj/                     the Vivado project (open it in the GUI to inspect)
#   cpl_versal.pdi            programmable device image (JTAG: program_jtag.tcl)
#   cpl_versal.xsa            hardware platform for Vitis / PetaLinux / Yocto
#   impl_timing_summary.rpt, impl_util_hier.rpt, impl_util.rpt, build_summary.txt
#
# STATUS: NOT run (no Vivado on the authoring machine); checked under tclsh
# with stub Vivado commands only. See README.md, "What is untested".
# ---------------------------------------------------------------------------

set here [file dirname [file normalize [info script]]]
source [file join $here common.tcl]
source [file join $here files.tcl]

cpl_check_args $argv {part out jobs to base pmc_ref_mhz pl0_mhz strategy}
set part     [cpl_getopt $argv part        $CPL_DEFAULT_PART CPL_PART]
set out      [file normalize [cpl_getopt $argv out [file join $here out] CPL_OUT]]
set jobs     [cpl_getopt $argv jobs        8 CPL_JOBS]
set stop_at  [cpl_getopt $argv to          pdi CPL_TO]
set strategy [cpl_getopt $argv strategy    "Vivado Implementation Defaults" CPL_IMPL_STRATEGY]
# consumed by bd_cips.tcl (which keeps any value already set)
set CPL_BASE_ADDR       [cpl_getopt $argv base        0xA4000000 CPL_BASE_ADDR]
set CPL_PMC_REF_CLK_MHZ [cpl_getopt $argv pmc_ref_mhz 33.3333    CPL_PMC_REF_CLK_MHZ]
set CPL_PL0_MHZ         [cpl_getopt $argv pl0_mhz     200        CPL_PL0_MHZ]

if {[lsearch -exact {bd synth pdi} $stop_at] < 0} {
    error "cpl: -to must be bd, synth or pdi (got '$stop_at')"
}
if {($CPL_BASE_ADDR % 0x200000) != 0} {
    error "cpl: -base $CPL_BASE_ADDR is not 2 MiB aligned"
}

set proj_name cpl_versal
set proj_dir  [file join $out proj]
set bd_name   cpl_bd

cpl_vivado_banner build_versal.tcl
puts "cpl: part=$part out=$out to=$stop_at base=$CPL_BASE_ADDR\
      pmc_ref=${CPL_PMC_REF_CLK_MHZ}MHz pl0=${CPL_PL0_MHZ}MHz"

cpl_require_part $part
cpl_check_rtl_files [list $CPL_BD_SHIM]
file mkdir $out

# ---- project -----------------------------------------------------------------
create_project -force $proj_name $proj_dir -part $part
set_property target_language Verilog [current_project]
add_files -norecurse -fileset sources_1 [concat $CPL_RTL_FILES [list $CPL_BD_SHIM]]
set_property file_type SystemVerilog [get_files -of_objects [get_filesets sources_1] *.sv]
update_compile_order -fileset sources_1

# ---- block design ------------------------------------------------------------
source [file join $here bd_cips.tcl]
set bd_file [cpl_create_bd $bd_name]
if {[llength $bd_file] == 0} { error "cpl: block design file not found" }

set wrapper [make_wrapper -files $bd_file -top]
add_files -norecurse -fileset sources_1 $wrapper
set_property top ${bd_name}_wrapper [get_filesets sources_1]
update_compile_order -fileset sources_1
generate_target all $bd_file

if {$stop_at eq "bd"} {
    puts "cpl: stopped after the block design (project: $proj_dir)"
    return
}

# ---- synthesis ---------------------------------------------------------------------
proc cpl_run_ok {run} {
    set prog [get_property PROGRESS [get_runs $run]]
    set stat [get_property STATUS   [get_runs $run]]
    puts "cpl: $run: $prog  $stat"
    if {$prog ne "100%" || [string match -nocase "*error*" $stat] \
            || [string match -nocase "*failed*" $stat]} {
        error "cpl: run $run did not complete ($stat) -- see\
               [get_property DIRECTORY [get_runs $run]]/runme.log"
    }
}

launch_runs synth_1 -jobs $jobs
wait_on_run synth_1
cpl_run_ok synth_1

if {$stop_at eq "synth"} {
    puts "cpl: stopped after synthesis (project: $proj_dir)"
    return
}

# ---- implementation + device image (Versal: write_device_image, not write_bitstream)
set_property strategy $strategy [get_runs impl_1]
launch_runs impl_1 -to_step write_device_image -jobs $jobs
wait_on_run impl_1
cpl_run_ok impl_1

set impl_dir [get_property DIRECTORY [get_runs impl_1]]
set pdi [file join $impl_dir ${bd_name}_wrapper.pdi]
if {![file isfile $pdi]} {
    set pdis [glob -nocomplain -directory $impl_dir *.pdi]
    if {[llength $pdis] != 1} {
        error "cpl: expected ${bd_name}_wrapper.pdi in $impl_dir, found: $pdis"
    }
    set pdi [lindex $pdis 0]
}
file copy -force $pdi [file join $out cpl_versal.pdi]

# ---- reports ---------------------------------------------------------------------------
open_run impl_1
report_timing_summary -max_paths 20 -file [file join $out impl_timing_summary.rpt]
report_utilization -file [file join $out impl_util.rpt]
report_utilization -hierarchical -hierarchical_depth 6 \
    -file [file join $out impl_util_hier.rpt]

set wns [get_property STATS.WNS [get_runs impl_1]]
set whs [get_property STATS.WHS [get_runs impl_1]]
set fh [open [file join $out build_summary.txt] w]
cpl_tee $fh "cpl: ==== build_versal summary ===="
cpl_tee $fh "cpl: part $part, cpl window $CPL_BASE_ADDR (2 MiB), PL clock ${CPL_PL0_MHZ} MHz"
cpl_tee $fh "cpl: WNS $wns ns, WHS $whs ns"
if {$wns eq "" || $wns < 0 || $whs eq "" || $whs < 0} {
    cpl_tee $fh "cpl: TIMING FAIL -- the .pdi was still written but may not\
                 work at ${CPL_PL0_MHZ} MHz. Lower -pl0_mhz or fix the paths in\
                 impl_timing_summary.rpt before programming."
} else {
    cpl_tee $fh "cpl: TIMING PASS"
}
cpl_tee $fh "cpl: PDI  [file join $out cpl_versal.pdi]"

# ---- hardware platform for Vitis / Linux -------------------------------------------
# -fixed: not an extensible (Vitis-acceleration) platform; -include_bit puts
# the PDI into the .xsa.
write_hw_platform -fixed -include_bit -force -file [file join $out cpl_versal.xsa]
cpl_tee $fh "cpl: XSA  [file join $out cpl_versal.xsa]"
close $fh

puts "cpl: build_versal.tcl done"
