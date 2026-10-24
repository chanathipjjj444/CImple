# ---------------------------------------------------------------------------
# xsdb_smoke.tcl -- first-light test of cpl_top over JTAG (after program_jtag.tcl)
#
#   xsdb xsdb_smoke.tcl                       built-in test (see below)
#   xsdb xsdb_smoke.tcl -hex my_test.hex      run an op file (format below)
#   tclsh xsdb_smoke.tcl -emit smoke.hex      only write the built-in test as
#                                             an op file (no hardware needed)
#
# Options (after the script name) / environment:
#   -base    / CPL_BASE_ADDR   cpl window, default 0xA4000000 (= bd_cips.tcl)
#   -url     / CPL_HW_URL      hw_server, default localhost:3121
#   -target  / CPL_XSDB_TARGET target name pattern(s) to try, in order; default
#                              {*A72*#0} {*R5*#0} {PMC}   (see "Target" below)
#   -cable   / CPL_JTAG_CABLE  substring of jtag_cable_name (several boards)
#   -hex     / CPL_SMOKE_HEX   op file to run instead of the built-in test
#   -emit    / -               write the built-in test to this file and exit
#   -timeout / CPL_TIMEOUT_MS  poll timeout, default 2000 ms
#
# Built-in test (exercises AXI regs, GBUF and IMEM windows, WLOAD, MATVEC
# RAW32, END; ARCH.md §3-§5):
#   1. ID == 0x43504C31 ("CPL1"). Nothing else is touched if this fails.
#   2. CTRL.CLEAR, STATUS must read idle.
#   3. GBUF[0x0000..0x07FF] = W, 32 partitions x 64 rows INT8, pstride 64,
#      W[p][r] = s8((5p + 3r + 1) mod 256)
#      GBUF[0x0800..0x083F] = x, x[r] = s8((37r + 11) mod 256)
#      GBUF[0x1000..0x107F] = 0xDEADBEEF (so a stale result cannot pass)
#   4. IMEM[0] WLOAD bank0 gaddr 0 pstride 64 p0 0 np 32
#      IMEM[1] MATVEC bank0 ntok 1 aaddr 0x800 oaddr 0x1000 cstride 4
#              tstride 128 ncols 32 acc NONE out RAW32
#      IMEM[2] END
#   5. PC_START = 0, CTRL.START, poll STATUS until done|error.
#   6. STATUS == done, PC == 2, INSTRET == 3, PASSES == 1, print CYCLES.
#   7. GBUF[0x1000 + 4p] == sum_r W[p][r]*x[r] (two's complement), p=0..31.
#   The expected values are computed here, independently of the RTL.
#
# Op file format (one op per line, '#' starts a comment, numbers are 0x-hex
# or decimal; OFF is the byte offset inside the 2 MiB window):
#   w OFF VALUE                 32-bit write
#   r OFF EXPECT [MASK]         read, require (rd & MASK) == (EXPECT & MASK)
#   p OFF MASK [TIMEOUT_MS]     poll until (rd & MASK) != 0
#   d OFF LABEL                 read and print
# The host-side XsdbBackend uses the same primitives this script uses:
#   mwr -force <addr> <val>   /   mrd -force -value <addr>
#
# Target ("which core does the access"): xsdb memory accesses go through a
# processor target, so the address is interpreted in that core's address map.
#   * "Cortex-A72 #0" (APU) is the natural choice: M_AXI_FPD is in its map.
#     After a JTAG-only PDI the A72s may be held in reset / powered down;
#     the script then tries "rst -processor" and "stop" before the access.
#   * "Cortex-R5 #0" (RPU) reaches the FPD PL aperture through the LPD->FPD
#     path in most configurations -- second choice.
#   * "PMC" is a last resort; whether its debug path reaches M_AXI_FPD is not
#     verified.
#   Run "targets" in xsdb to see the real names on your board.
# Caveats:
#   * NEVER access the window before the PDI with cpl_top is loaded: a read
#     to an aperture with no responding slave can hang the PS interconnect
#     (recovery = power cycle / POR).
#   * An access that gets SLVERR from cpl_axil (unaligned / unmapped / IMEM
#     or GBUF while busy) is reported by xsdb as a memory access error; it is
#     not a crash.
#   * JTAG is slow (~ms per access): fine for this test, use the Linux
#     DevMemBackend for real workloads.
#
# STATUS: the op engine and the built-in test were checked under tclsh with
# a Python model of the ARCH.md semantics standing in for mrd/mwr; never
# run against hardware or real xsdb.
# ---------------------------------------------------------------------------

set here [file dirname [file normalize [info script]]]
source [file join $here common.tcl]

set CPL_ID_VALUE 0x43504C31
set CPL_GBUF     0x100000
set CPL_IMEM     0x010000

# ---------------------------------------------------------------------------
# numbers / ops
# ---------------------------------------------------------------------------
# Strict number parser for op files: 0x-hex or decimal without leading zero
# (Tcl 8.6 would read "010" as octal). Never passes file text to expr.
proc cpl_num {tok} {
    if {[regexp {^0[xX][0-9a-fA-F]+$} $tok]} {
        scan [string range $tok 2 end] %llx v
        return $v
    }
    if {[regexp {^(0|[1-9][0-9]*)$} $tok]} {
        scan $tok %lld v
        return $v
    }
    error "cpl: bad number '$tok'"
}

proc cpl_hex32 {v} { return [format 0x%08X [expr {$v & 0xFFFFFFFF}]] }

proc cpl_s8 {b} { return [expr {$b >= 128 ? $b - 256 : $b}] }

# Built-in test as an op list.
proc cpl_builtin_ops {} {
    global CPL_ID_VALUE CPL_GBUF CPL_IMEM
    set ops {}
    set gw  0x0000
    set gx  0x0800
    set go  0x1000

    lappend ops [list r 0x000 $CPL_ID_VALUE 0xFFFFFFFF "ID"]
    lappend ops [list w 0x004 0x2 "CTRL.CLEAR"]
    lappend ops [list r 0x008 0x0 0x0000FF07 "STATUS idle after CLEAR"]

    # weights: byte at gw + 64p + r
    set W {}
    for {set p 0} {$p < 32} {incr p} {
        set row {}
        for {set r 0} {$r < 64} {incr r} {
            lappend row [expr {(5*$p + 3*$r + 1) & 0xFF}]
        }
        lappend W $row
        for {set r 0} {$r < 64} {incr r 4} {
            set v 0
            for {set k 0} {$k < 4} {incr k} {
                set v [expr {$v | ([lindex $row [expr {$r + $k}]] << (8*$k))}]
            }
            lappend ops [list w [expr {$CPL_GBUF + $gw + 64*$p + $r}] $v ""]
        }
    }
    # activations
    set X {}
    for {set r 0} {$r < 64} {incr r} { lappend X [expr {(37*$r + 11) & 0xFF}] }
    for {set r 0} {$r < 64} {incr r 4} {
        set v 0
        for {set k 0} {$k < 4} {incr k} {
            set v [expr {$v | ([lindex $X [expr {$r + $k}]] << (8*$k))}]
        }
        lappend ops [list w [expr {$CPL_GBUF + $gx + $r}] $v ""]
    }
    # poison the output area
    for {set p 0} {$p < 32} {incr p} {
        lappend ops [list w [expr {$CPL_GBUF + $go + 4*$p}] 0xDEADBEEF ""]
    }
    # GBUF read-back sanity (first weight word, last activation word)
    set w00 [expr {[lindex $W 0 0] | ([lindex $W 0 1] << 8) | ([lindex $W 0 2] << 16) | ([lindex $W 0 3] << 24)}]
    lappend ops [list r [expr {$CPL_GBUF + $gw}] $w00 0xFFFFFFFF "GBUF read-back"]

    # program (ARCH.md §4)
    set i_wload  [list 0x00000001 $gw 64 [expr {(32 << 8) | 0}] 0 0 0 0]
    set w0_mv    [expr {2 | (0 << 4) | (0 << 5) | (4 << 7) | (1 << 16)}]
    set i_matvec [list $w0_mv $gx 64 $go [expr {(4 << 16) | 128}] \
                      [expr {(32 << 24) | (0 << 16) | 1}] 0 0]
    set i_end    [list 0 0 0 0 0 0 0 0]
    set idx 0
    foreach ins [list $i_wload $i_matvec $i_end] {
        for {set w 0} {$w < 8} {incr w} {
            lappend ops [list w [expr {$CPL_IMEM + 32*$idx + 4*$w}] [lindex $ins $w] ""]
        }
        incr idx
    }
    lappend ops [list r [expr {$CPL_IMEM + 32}] $w0_mv 0xFFFFFFFF "IMEM read-back"]

    # run
    lappend ops [list w 0x00C 0 "PC_START"]
    lappend ops [list w 0x004 0x1 "CTRL.START"]
    lappend ops [list p 0x008 0x6 -1 "wait done|error"]
    lappend ops [list r 0x008 0x2 0x0000FF07 "STATUS done, no error"]
    lappend ops [list r 0x010 2 0xFF "PC at END"]
    lappend ops [list r 0x018 3 0xFFFFFFFF "INSTRET"]
    lappend ops [list r 0x01C 1 0xFFFFFFFF "PASSES"]
    lappend ops [list d 0x014 "CYCLES"]
    lappend ops [list d 0x020 "CFG"]

    # expected RAW32 outputs
    for {set p 0} {$p < 32} {incr p} {
        set acc 0
        for {set r 0} {$r < 64} {incr r} {
            set acc [expr {$acc + [cpl_s8 [lindex $W $p $r]] * [cpl_s8 [lindex $X $r]]}]
        }
        lappend ops [list r [expr {$CPL_GBUF + $go + 4*$p}] [expr {$acc & 0xFFFFFFFF}] \
                         0xFFFFFFFF "RAW32 p=$p ($acc)"]
    }
    return $ops
}

proc cpl_write_ops {ops file} {
    set fh [open $file w]
    puts $fh "# cpl smoke test ops (generated by xsdb_smoke.tcl -emit); see its header"
    foreach op $ops {
        set c [lindex $op 0]
        switch -- $c {
            w { puts $fh [format "w 0x%06X %s%s" [lindex $op 1] [cpl_hex32 [lindex $op 2]] \
                              [expr {[lindex $op 3] ne "" ? "  # [lindex $op 3]" : ""}]] }
            r { puts $fh [format "r 0x%06X %s %s  # %s" [lindex $op 1] [cpl_hex32 [lindex $op 2]] \
                              [cpl_hex32 [lindex $op 3]] [lindex $op 4]] }
            p { puts $fh [format "p 0x%06X %s  # %s" [lindex $op 1] [cpl_hex32 [lindex $op 2]] \
                              [lindex $op 4]] }
            d { puts $fh [format "d 0x%06X %s" [lindex $op 1] [lindex $op 2]] }
        }
    }
    close $fh
}

proc cpl_read_ops {file} {
    set fh [open $file r]
    set ops {}
    set ln 0
    while {[gets $fh line] >= 0} {
        incr ln
        set hash [string first "#" $line]
        if {$hash >= 0} { set line [string range $line 0 [expr {$hash - 1}]] }
        set t [regexp -all -inline {\S+} $line]
        if {[llength $t] == 0} { continue }
        set c [lindex $t 0]
        if {[catch {
            switch -- $c {
                w { if {[llength $t] != 3} { error "w OFF VALUE" }
                    lappend ops [list w [cpl_num [lindex $t 1]] [cpl_num [lindex $t 2]] ""] }
                r { if {[llength $t] < 3 || [llength $t] > 4} { error "r OFF EXPECT ?MASK?" }
                    set m [expr {[llength $t] == 4 ? [cpl_num [lindex $t 3]] : 0xFFFFFFFF}]
                    lappend ops [list r [cpl_num [lindex $t 1]] [cpl_num [lindex $t 2]] $m "line $ln"] }
                p { if {[llength $t] < 3 || [llength $t] > 4} { error "p OFF MASK ?TIMEOUT_MS?" }
                    set to [expr {[llength $t] == 4 ? [cpl_num [lindex $t 3]] : -1}]
                    lappend ops [list p [cpl_num [lindex $t 1]] [cpl_num [lindex $t 2]] $to "line $ln"] }
                d { if {[llength $t] < 2} { error "d OFF ?LABEL?" }
                    lappend ops [list d [cpl_num [lindex $t 1]] [join [lrange $t 2 end] " "]] }
                default { error "unknown op '$c'" }
            }
        } e]} {
            close $fh
            error "cpl: $file:$ln: $e"
        }
    }
    close $fh
    return $ops
}

# ---------------------------------------------------------------------------
# hardware access (xsdb). Kept in two procs so a test harness can replace them.
# ---------------------------------------------------------------------------
proc cpl_wr {addr val} { mwr -force [format 0x%lX $addr] [cpl_hex32 $val] }
proc cpl_rd {addr} {
    return [expr {[lindex [mrd -force -value [format 0x%lX $addr]] 0] & 0xFFFFFFFF}]
}

# Returns the number of failed checks.
proc cpl_run_ops {base ops timeout_ms} {
    set fails 0
    set checks 0
    foreach op $ops {
        set c [lindex $op 0]
        set a [expr {$base + [lindex $op 1]}]
        switch -- $c {
            w { cpl_wr $a [lindex $op 2] }
            r {
                set exp [lindex $op 2]
                set m   [lindex $op 3]
                set got [cpl_rd $a]
                incr checks
                if {($got & $m) != ($exp & $m)} {
                    incr fails
                    puts "cpl: FAIL [lindex $op 4]: @[format 0x%lX $a] got\
                          [cpl_hex32 $got] expected [cpl_hex32 $exp] (mask [cpl_hex32 $m])"
                    if {[lindex $op 1] == 0} {
                        puts "cpl: ID mismatch -- wrong base address, PDI not loaded,\
                              or wrong target; stopping before touching anything else"
                        return $fails
                    }
                } elseif {[lindex $op 4] ne "" && ![string match "RAW32*" [lindex $op 4]]} {
                    puts "cpl: ok   [lindex $op 4] = [cpl_hex32 $got]"
                }
            }
            p {
                set to [lindex $op 3]
                if {$to < 0} { set to $timeout_ms }
                set t0 [clock milliseconds]
                while {1} {
                    set got [cpl_rd $a]
                    if {($got & [lindex $op 2]) != 0} { break }
                    if {[clock milliseconds] - $t0 > $to} {
                        incr fails
                        puts "cpl: FAIL [lindex $op 4]: timeout after $to ms,\
                              last value [cpl_hex32 $got]"
                        return $fails
                    }
                    after 1
                }
                puts "cpl: ok   [lindex $op 4] (STATUS [cpl_hex32 $got])"
                if {($got & 0x4) != 0} {
                    puts "cpl: device reports error, err_code = [expr {($got >> 8) & 0xFF}]"
                }
            }
            d { puts "cpl: info [lindex $op 2] = [cpl_hex32 [cpl_rd $a]] ([cpl_rd $a])" }
        }
    }
    puts "cpl: $checks checks, $fails failed"
    return $fails
}

# ---------------------------------------------------------------------------
# xsdb target selection
# ---------------------------------------------------------------------------
proc cpl_select_target {url patterns cable base} {
    global CPL_ID_VALUE
    connect -url tcp:$url
    foreach pat $patterns {
        set filt "name =~ \"$pat\""
        if {$cable ne ""} { append filt " && jtag_cable_name =~ \"*${cable}*\"" }
        if {[catch {targets -set -nocase -filter $filt} e]} {
            puts "cpl: target '$pat' not found ($e)"
            continue
        }
        puts "cpl: trying target '$pat'"
        catch {stop}
        if {[catch {cpl_rd $base} id]} {
            puts "cpl:   access failed ($id); resetting the core into halt and retrying"
            if {[catch {rst -processor -stop}]} { catch {rst -processor} }
            catch {stop}
            if {[catch {cpl_rd $base} id]} {
                puts "cpl:   still failing ($id)"
                continue
            }
        }
        puts "cpl:   ID read = [cpl_hex32 $id] via '$pat'"
        return $pat
    }
    puts stderr "cpl: no target could read the cpl window. Available targets:"
    catch {puts stderr [targets]}
    error "cpl: no usable xsdb target (try -target <name pattern>)"
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
cpl_check_args $argv {base url target cable hex emit timeout}
set base    [cpl_num [cpl_getopt $argv base 0xA4000000 CPL_BASE_ADDR]]
set url     [cpl_getopt $argv url localhost:3121 CPL_HW_URL]
set tpats   [cpl_getopt $argv target {{*A72*#0} {*R5*#0} {PMC}} CPL_XSDB_TARGET]
set cable   [cpl_getopt $argv cable "" CPL_JTAG_CABLE]
set hexfile [cpl_getopt $argv hex "" CPL_SMOKE_HEX]
set emit    [cpl_getopt $argv emit ""]
set tmo     [cpl_num [cpl_getopt $argv timeout 2000 CPL_TIMEOUT_MS]]

if {$emit ne ""} {
    cpl_write_ops [cpl_builtin_ops] $emit
    puts "cpl: wrote $emit"
    return
}

if {$hexfile ne ""} {
    set ops [cpl_read_ops $hexfile]
} else {
    set ops [cpl_builtin_ops]
}

# CPL_SMOKE_NO_CONNECT: test hook -- the harness provides cpl_rd / cpl_wr.
if {![info exists ::env(CPL_SMOKE_NO_CONNECT)]} {
    cpl_select_target $url $tpats $cable $base
}
set nfail [cpl_run_ops $base $ops $tmo]
if {$nfail == 0} {
    puts "cpl: SMOKE PASS"
} else {
    puts "cpl: SMOKE FAIL ($nfail)"
    if {![info exists ::env(CPL_SMOKE_NO_CONNECT)]} { exit 1 }
}
