# ---------------------------------------------------------------------------
# program_jtag.tcl -- program cpl_versal.pdi into the Versal over JTAG
#
# Runs under either tool (it detects which one it is in):
#
#   Vivado Hardware Manager:
#     vivado -mode batch -source program_jtag.tcl -tclargs -pdi out/cpl_versal.pdi
#   xsdb (Vitis / standalone "Vivado Lab Edition" install also ships hw_server):
#     xsdb program_jtag.tcl -pdi out/cpl_versal.pdi
#
# Options (Vivado: after -tclargs; xsdb: after the script name) / environment:
#   -pdi    / CPL_PDI      default fpga/vivado/out/cpl_versal.pdi
#   -url    / CPL_HW_URL   hw_server, default localhost:3121 (start it with
#                          "hw_server" on the machine the JTAG cable is on)
#   -cable  / CPL_JTAG_CABLE  substring of the cable/target name to use when
#                          more than one board is connected (default: first)
#   -device / CPL_HW_DEVICE   device name pattern, default xcvp1202*
#
# READ README.md "BoostX / MangoBoost caveats" FIRST: programming a new PDI
# replaces the vendor shell, the PCIe link of the card goes down, and the host
# may hang or crash. Have a way to power-cycle and restore the vendor image.
#
# STATUS: NOT run (no hardware / Vivado on the authoring machine).
# ---------------------------------------------------------------------------

set here [file dirname [file normalize [info script]]]
source [file join $here common.tcl]

cpl_check_args $argv {pdi url cable device}
set pdi    [file normalize [cpl_getopt $argv pdi [file join $here out cpl_versal.pdi] CPL_PDI]]
set url    [cpl_getopt $argv url    localhost:3121 CPL_HW_URL]
set cable  [cpl_getopt $argv cable  "" CPL_JTAG_CABLE]
set devpat [cpl_getopt $argv device xcvp1202* CPL_HW_DEVICE]

if {![file isfile $pdi]} {
    error "cpl: PDI '$pdi' not found (run build_versal.tcl first)"
}

# ===========================================================================
# Vivado Hardware Manager
# ===========================================================================
proc cpl_program_vivado {pdi url cable devpat} {
    open_hw_manager
    connect_hw_server -url $url
    set targets [get_hw_targets -quiet]
    if {$cable ne ""} {
        set targets [get_hw_targets -quiet *${cable}*]
    }
    if {[llength $targets] == 0} {
        error "cpl: no JTAG target on $url (cable '$cable'). Is the cable\
               plugged in, the card powered, and hw_server running?"
    }
    puts "cpl: JTAG targets: $targets"
    current_hw_target [lindex $targets 0]
    open_hw_target

    # A Versal shows up as the device (e.g. xcvp1202_1) plus possibly an
    # arm_dap_* entry; program the FPGA device, not the DAP.
    set devs [get_hw_devices -quiet $devpat]
    if {[llength $devs] == 0} {
        error "cpl: no hw_device matching '$devpat' on this target; devices\
               found: [get_hw_devices -quiet]. Check the part on the card and\
               pass -device <pattern>."
    }
    set dev [lindex $devs 0]
    puts "cpl: programming $dev with $pdi"
    current_hw_device $dev
    refresh_hw_device -update_hw_probes false $dev
    set_property PROGRAM.FILE $pdi $dev
    # no debug cores in this design -> no probes file
    set_property PROBES.FILE {} $dev
    program_hw_devices $dev
    refresh_hw_device $dev
    puts "cpl: programmed. Next: xsdb xsdb_smoke.tcl"
    close_hw_target
    disconnect_hw_server
    close_hw_manager
}

# ===========================================================================
# xsdb
# ===========================================================================
proc cpl_program_xsdb {pdi url cable devpat} {
    connect -url tcp:$url
    # The top-level Versal target is named "Versal xcvp1202" (one level above
    # PMC/PSM/APU/RPU/PL). "device program" must be issued with it selected.
    set filt {name =~ "Versal*"}
    if {$cable ne ""} {
        set filt "name =~ \"Versal*\" && jtag_cable_name =~ \"*${cable}*\""
    }
    if {[catch {targets -set -nocase -filter $filt} e]} {
        puts stderr "cpl: could not select the Versal target ($e). Targets:"
        puts stderr [targets]
        error "cpl: no Versal target"
    }
    puts "cpl: programming $pdi (device program) ..."
    device program $pdi
    puts "cpl: programmed. Next: xsdb xsdb_smoke.tcl"
}

if {[llength [info commands open_hw_manager]] > 0} {
    cpl_program_vivado $pdi $url $cable $devpat
} elseif {[llength [info commands device]] > 0 && [llength [info commands targets]] > 0} {
    cpl_program_xsdb $pdi $url $cable $devpat
} else {
    error "cpl: run this script from Vivado (-mode batch -source) or xsdb"
}
