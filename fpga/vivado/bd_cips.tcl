# ---------------------------------------------------------------------------
# bd_cips.tcl -- IP-integrator block design: CIPS -> SmartConnect -> cpl_top
#
#   versal_cips (PS on, JTAG debug, PL clock pl0_ref_clk = 200 MHz, pl0_resetn,
#                M_AXI_FPD master; no DDR, no NoC, no CPM/PCIe)
#        | M_AXI_FPD (AXI4, 128b)          pl0_ref_clk ----+---- all aclk
#   axi_smartconnect (AXI4 -> AXI4-Lite 32b, 1 SI / 1 MI)   |
#        | M00_AXI                                          |
#   cpl_top_bd (module reference, s_axil, 2 MiB @ CPL_BASE_ADDR)
#   proc_sys_reset: ext_reset_in = pl0_resetn, interconnect_aresetn -> SC,
#                   peripheral_aresetn -> cpl rst_ni
#   cpl irq_o (level high) -> CIPS pl_ps_irq0 when this CIPS version accepts
#                   PS_IRQ_USAGE; otherwise left open (software polls STATUS)
#
# Usage:
#   * from build_versal.tcl (normal), or
#   * in the Vivado GUI Tcl console with the project already open and the RTL
#     + cpl_top_bd.v added:  source fpga/vivado/bd_cips.tcl ; cpl_create_bd
#
# STATUS: written against Vivado 2024.2 / 2025.1 (CIPS 3.4 / 3.5 era); NOT run
# (no Vivado on the authoring machine). The CIPS PS_PMC_CONFIG key names
# below are the ones Vivado 2022.1+ writes in exported BD scripts; they are
# version-dependent. If CIPS rejects a key or a pin is missing after
# configuration, open the CIPS GUI once (double-click the cell), set the
# options listed in README.md by hand, and run write_bd_tcl to see the key
# names your version uses.
# ---------------------------------------------------------------------------

# ===========================================================================
# BOARD-SPECIFIC SETTINGS -- check every value against the BoostX documentation
# (MangoBoost hardware user guide / schematic, or the vendor's reference
# Vivado project: open it and read the CIPS configuration). Each may be
# preset by the caller (build_versal.tcl sets them from -tclargs / env).
# ===========================================================================

# PMC reference clock (REF_CLK pin of the PMC) in MHz. 33.3333 is the AMD
# evaluation-board value (VCK190/VPK120). BoostX: read the oscillator on the
# PMC REF_CLK pin in the schematic or the vendor project's CIPS
# "PMC_REF_CLK_FREQMHZ". A wrong value makes every PS/PL clock wrong.
if {![info exists CPL_PMC_REF_CLK_MHZ]} { set CPL_PMC_REF_CLK_MHZ 33.3333 }

# PL clock 0 frequency (MHz) feeding cpl_top. 200 = the timing target.
if {![info exists CPL_PL0_MHZ]}         { set CPL_PL0_MHZ 200 }

# Base address of the 2 MiB cpl window in the M_AXI_FPD aperture
# (Versal: 0xA400_0000 - 0xAFFF_FFFF, 192 MiB, plus a high aperture).
# Must be 2 MiB aligned. The host driver (DevMemBackend/XsdbBackend) and
# xsdb_smoke.tcl must use the same value.
if {![info exists CPL_BASE_ADDR]}       { set CPL_BASE_ADDR 0xA4000000 }
if {![info exists CPL_RANGE]}           { set CPL_RANGE 2M }

# Debug / boot: JTAG only for development [A]. A boot device (QSPI/OSPI/SD/
# eMMC) is only needed to boot this PDI without a cable -- do NOT overwrite
# the card's flash, it holds the MangoBoost image.
if {![info exists CPL_DEBUG_MODE]}      { set CPL_DEBUG_MODE JTAG }

# Extra PS_PMC_CONFIG key/value pairs (board MIO, UART, ...), appended last
# so they override the generic values below. Example (VCK190-style UART0 on
# PMC MIO 42..43, NOT verified for BoostX):
#   set CPL_PMC_EXTRA {PS_UART0_PERIPHERAL {{ENABLE 1} {IO {PMC_MIO 42 .. 43}}}}
if {![info exists CPL_PMC_EXTRA]}       { set CPL_PMC_EXTRA {} }

# ===========================================================================

# Newest installed version of an IP (VLNV without version).
proc cpl_latest_vlnv {vlnv_prefix} {
    set defs [get_ipdefs -all -quiet ${vlnv_prefix}:*]
    if {[llength $defs] == 0} {
        error "cpl: IP '$vlnv_prefix' is not available in this Vivado install"
    }
    return [lindex [lsort -dictionary $defs] end]
}

proc cpl_require_pin {path {intf 0}} {
    if {$intf} {
        set o [get_bd_intf_pins -quiet $path]
    } else {
        set o [get_bd_pins -quiet $path]
    }
    if {[llength $o] == 0} {
        error "cpl: block-design pin '$path' does not exist. If it is a CIPS\
               pin, the PS_PMC_CONFIG keys in bd_cips.tcl were not accepted by\
               this CIPS version: configure CIPS in the GUI (see README.md,\
               'CIPS settings') and compare with write_bd_tcl output."
    }
    return $o
}

# Sets a CONFIG.* property if this IP version has it; warns otherwise.
proc cpl_try_config {cell key value} {
    if {[catch {set_property CONFIG.$key $value $cell} e]} {
        puts "cpl: WARNING: [get_property NAME $cell]: CONFIG.$key not accepted\
              (version-dependent, continuing): $e"
        return 0
    }
    return 1
}

proc cpl_create_bd {{bd_name cpl_bd}} {
    global CPL_PMC_REF_CLK_MHZ CPL_PL0_MHZ CPL_BASE_ADDR CPL_RANGE \
           CPL_DEBUG_MODE CPL_PMC_EXTRA

    create_bd_design $bd_name
    current_bd_design $bd_name

    # ---- CIPS ----------------------------------------------------------------
    set cips_vlnv [cpl_latest_vlnv xilinx.com:ip:versal_cips]
    puts "cpl: using $cips_vlnv"
    set cips [create_bd_cell -type ip -vlnv $cips_vlnv cips]

    # Top-level CIPS mode properties (present in CIPS 3.x; each tried
    # separately because the set differs between versions).
    cpl_try_config $cips DESIGN_MODE             1
    cpl_try_config $cips BOOT_MODE               Custom
    cpl_try_config $cips CLOCK_MODE              Custom
    cpl_try_config $cips DDR_MEMORY_MODE         Custom
    cpl_try_config $cips DEBUG_MODE              $CPL_DEBUG_MODE
    cpl_try_config $cips IO_CONFIG_MODE          Custom
    cpl_try_config $cips PS_BOARD_INTERFACE      Custom
    cpl_try_config $cips PS_PL_CONNECTIVITY_MODE Custom

    # PS/PMC configuration dictionary (Vivado 2022.1+ key names) [A]:
    #   DESIGN_MODE 1                 PS + PL design (not PL-only)
    #   PS_USE_M_AXI_FPD 1            the FPD AXI master into the PL
    #   PS_USE_PMCPL_CLK0 1           pl0_ref_clk
    #   PMC_CRP_PL0_REF_CTRL_FREQMHZ  its frequency
    #   PS_NUM_FABRIC_RESETS 1        pl0_resetn
    #   no PMC_USE_PMC_NOC_AXI0 / PS_USE_FPD_CCI_NOC: no NoC, no DDR
    set pmc_cfg [list \
        DESIGN_MODE                  1 \
        BOOT_MODE                    Custom \
        CLOCK_MODE                   Custom \
        DDR_MEMORY_MODE              Custom \
        DEBUG_MODE                   $CPL_DEBUG_MODE \
        IO_CONFIG_MODE               Custom \
        PS_BOARD_INTERFACE           Custom \
        PS_PL_CONNECTIVITY_MODE      Custom \
        PMC_REF_CLK_FREQMHZ          $CPL_PMC_REF_CLK_MHZ \
        PS_USE_PMCPL_CLK0            1 \
        PMC_CRP_PL0_REF_CTRL_FREQMHZ $CPL_PL0_MHZ \
        PS_NUM_FABRIC_RESETS         1 \
        PS_USE_M_AXI_FPD             1 \
        ]
    foreach {k v} $CPL_PMC_EXTRA { lappend pmc_cfg $k $v }
    if {[catch {set_property CONFIG.PS_PMC_CONFIG $pmc_cfg $cips} e]} {
        error "cpl: CIPS rejected CONFIG.PS_PMC_CONFIG ($e). This key set is\
               version-dependent -- see the header of bd_cips.tcl."
    }

    cpl_require_pin cips/pl0_ref_clk
    cpl_require_pin cips/pl0_resetn
    cpl_require_pin cips/m_axi_fpd_aclk
    cpl_require_pin cips/M_AXI_FPD 1

    # ---- SmartConnect: AXI4 128b -> AXI4-Lite 32b ------------------------------
    set sc [create_bd_cell -type ip \
                -vlnv [cpl_latest_vlnv xilinx.com:ip:smartconnect] sc]
    set_property -dict [list CONFIG.NUM_SI 1 CONFIG.NUM_MI 1 CONFIG.NUM_CLKS 1] $sc

    # ---- reset synchroniser ------------------------------------------------------
    set rst [create_bd_cell -type ip \
                 -vlnv [cpl_latest_vlnv xilinx.com:ip:proc_sys_reset] rst_pl0]
    # pl0_resetn is active low; IP integrator normally propagates this,
    # setting it explicitly is harmless where the parameter exists.
    cpl_try_config $rst C_EXT_RESET_HIGH 0

    # ---- the accelerator (module reference on the Verilog shim) -------------
    if {[catch {create_bd_cell -type module -reference cpl_top_bd cpl} e]} {
        error "cpl: could not create the module reference 'cpl_top_bd' ($e).\
               Add fpga/rtl/*.sv and fpga/vivado/cpl_top_bd.v to sources_1 and\
               run update_compile_order first."
    }
    cpl_require_pin cpl/s_axil 1
    cpl_require_pin cpl/clk_i
    cpl_require_pin cpl/rst_ni
    cpl_require_pin cpl/irq_o

    # ---- interrupt: cpl irq_o -> CIPS PL-to-PS IRQ 0 [A] ------------------------
    # PS_IRQ_USAGE (CIPS 3.x key, as written by write_bd_tcl) enables
    # pl_ps_irq0. Applied as a second PS_PMC_CONFIG set so that a CIPS
    # version that rejects the key keeps the configuration above; then the
    # interrupt is simply not wired and software polls STATUS (the driver
    # does that anyway: irq_o = STATUS.done | STATUS.error).
    set irq_cfg [list PS_IRQ_USAGE {{CH0 1} {CH1 0} {CH2 0} {CH3 0} {CH4 0} {CH5 0} {CH6 0} {CH7 0} {CH8 0} {CH9 0} {CH10 0} {CH11 0} {CH12 0} {CH13 0} {CH14 0} {CH15 0}}]
    if {[catch {set_property CONFIG.PS_PMC_CONFIG [concat $pmc_cfg $irq_cfg] $cips} e]} {
        puts "cpl: WARNING: CIPS did not accept PS_IRQ_USAGE ($e)"
    }
    if {[llength [get_bd_pins -quiet cips/pl_ps_irq0]] > 0} {
        connect_bd_net [get_bd_pins cpl/irq_o] [get_bd_pins cips/pl_ps_irq0]
        puts "cpl: cpl/irq_o -> cips/pl_ps_irq0 (level high)"
    } else {
        puts "cpl: WARNING: cips/pl_ps_irq0 not present: cpl/irq_o is left\
              unconnected, software must poll STATUS"
    }

    # ---- clocks and resets -----------------------------------------------------
    connect_bd_net [get_bd_pins cips/pl0_ref_clk] \
                   [get_bd_pins cips/m_axi_fpd_aclk] \
                   [get_bd_pins sc/aclk] \
                   [get_bd_pins rst_pl0/slowest_sync_clk] \
                   [get_bd_pins cpl/clk_i]
    connect_bd_net [get_bd_pins cips/pl0_resetn] [get_bd_pins rst_pl0/ext_reset_in]
    connect_bd_net [get_bd_pins rst_pl0/interconnect_aresetn] [get_bd_pins sc/aresetn]
    connect_bd_net [get_bd_pins rst_pl0/peripheral_aresetn] [get_bd_pins cpl/rst_ni]

    # ---- AXI -----------------------------------------------------------------------
    connect_bd_intf_net [get_bd_intf_pins cips/M_AXI_FPD] [get_bd_intf_pins sc/S00_AXI]
    connect_bd_intf_net [get_bd_intf_pins sc/M00_AXI]     [get_bd_intf_pins cpl/s_axil]

    # ---- address map: 2 MiB at CPL_BASE_ADDR in M_AXI_FPD's space --------------
    set space [get_bd_addr_spaces -quiet cips/M_AXI_FPD]
    if {[llength $space] == 0} {
        error "cpl: address space cips/M_AXI_FPD not found"
    }
    set slave_seg [get_bd_addr_segs -quiet cpl/s_axil/reg0]
    if {[llength $slave_seg] == 0} {
        set slave_seg [get_bd_addr_segs -quiet -of_objects [get_bd_intf_pins cpl/s_axil]]
    }
    if {[llength $slave_seg] != 1} {
        error "cpl: expected one address segment on cpl/s_axil, got '$slave_seg'"
    }
    assign_bd_address -offset $CPL_BASE_ADDR -range $CPL_RANGE \
        -target_address_space $space $slave_seg -force

    # Read the assignment back and fail loudly if it did not land where the
    # software expects it.
    set mseg [get_bd_addr_segs -quiet -of_objects $space]
    set ok 0
    foreach s $mseg {
        # the GUI shows 0x0000_A400_0000-style values; strip any '_'
        set off [string map {_ ""} [get_property OFFSET $s]]
        set rng [get_property RANGE $s]
        puts "cpl: address segment $s : offset $off range $rng"
        if {[string is wideinteger -strict $off] && $off == $CPL_BASE_ADDR} { set ok 1 }
    }
    if {!$ok} {
        error "cpl: cpl window is not at $CPL_BASE_ADDR (got: $mseg)"
    }

    catch {regenerate_bd_layout}
    validate_bd_design
    save_bd_design
    return [get_files -quiet ${bd_name}.bd]
}
