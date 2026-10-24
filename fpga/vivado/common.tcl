# ---------------------------------------------------------------------------
# common.tcl -- small helpers shared by the Vivado scripts in fpga/vivado
#
# Option handling: every script takes "-name value" pairs after -tclargs,
#   vivado -mode batch -source synth_ooc.tcl -tclargs -part xcvp1202-... -jobs 8
# and most options can also come from an environment variable (CPL_PART,
# CPL_OUT, ...). Priority: -tclargs > environment > built-in default.
# Unknown options are an error (catches typos such as "-prat").
#
# Plain Tcl 8.5 except cpl_require_part / cpl_vivado_banner, which call
# Vivado commands and are only used from Vivado.
# ---------------------------------------------------------------------------

# Default part [A]: the user plans against VP1202 "for now"; the BoostX
# card's real part/package/speed grade is NOT confirmed. Read it from the
# card (Vivado Hardware Manager shows the device; the package/speed grade are
# on the vendor documentation / the chip marking) and override with -part or
# CPL_PART.
set CPL_DEFAULT_PART xcvp1202-vsva2785-2MP-e-S

# cpl_check_args argv {known option names without the dash}
proc cpl_check_args {argv_list known} {
    if {[llength $argv_list] % 2 != 0} {
        error "cpl: options must be '-name value' pairs, got: $argv_list"
    }
    foreach {k v} $argv_list {
        if {[string index $k 0] ne "-"} {
            error "cpl: expected an option starting with '-', got '$k'"
        }
        set name [string range $k 1 end]
        if {[lsearch -exact $known $name] < 0} {
            error "cpl: unknown option '$k' (known: -[join $known { -}])"
        }
    }
}

# cpl_getopt argv name default ?ENV_VAR?
proc cpl_getopt {argv_list name default {env_name ""}} {
    foreach {k v} $argv_list {
        if {$k eq "-$name"} { return $v }
    }
    if {$env_name ne "" && [info exists ::env($env_name)] \
            && $::env($env_name) ne ""} {
        return $::env($env_name)
    }
    return $default
}

# Fails with a clear message when the part is unknown to this Vivado install.
# get_parts returns an empty list (plus a warning, silenced by -quiet) for a
# part that is not installed or not licensed.
proc cpl_require_part {part} {
    if {[llength [get_parts -quiet $part]] == 0} {
        set similar [get_parts -quiet xcvp1202*]
        set msg "cpl: part '$part' is not available in this Vivado installation.\n"
        append msg "  Possible causes:\n"
        append msg "   * Versal Premium device support is not installed\n"
        append msg "     (Help > Add Design Tools or Devices, or re-run the installer\n"
        append msg "      and tick 'Versal Premium' under Devices);\n"
        append msg "   * the edition/licence does not cover this device: VP1202 needs\n"
        append msg "     an Enterprise (paid) licence, the free Standard edition\n"
        append msg "     does not include it;\n"
        append msg "   * the part string is wrong (package / speed grade).\n"
        if {[llength $similar] > 0} {
            append msg "  Installed xcvp1202 parts: [join [lsort $similar] {, }]\n"
            append msg "  Re-run with -tclargs -part <one of them>  (or set CPL_PART)."
        } else {
            append msg "  No xcvp1202* part is installed at all."
        }
        puts stderr $msg
        exit 2
    }
}

proc cpl_vivado_banner {script} {
    set v "unknown"
    catch {set v [version -short]}
    puts "cpl: $script running under Vivado $v"
    # Tested-against versions (by review only, see README): 2024.2, 2025.1.
    if {![string match "2024.2*" $v] && ![string match "2025.*" $v]} {
        puts "cpl: WARNING: scripts were written for Vivado 2024.2 / 2025.1;\
              CIPS CONFIG keys in particular are version-dependent."
    }
}

# Write one line both to stdout and to an open report channel.
proc cpl_tee {fh line} {
    puts $line
    puts $fh $line
}
