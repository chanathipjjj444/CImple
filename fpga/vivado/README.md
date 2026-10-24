# fpga/vivado — build and flash flow for the Versal VP1202

This directory takes the SystemVerilog in `fpga/rtl` (contract: `fpga/docs/ARCH.md`)
to a programmable device image (`.pdi`) for an AMD Versal Premium VP1202
(`xcvp1202-vsva2785-2MP-e-S` by default), loads it over JTAG, and runs a first
test on the hardware.

> **Status: none of these scripts has been run in Vivado or on hardware.**
> They were written from the Vivado 2024.2 / 2025.1 documentation on a machine
> that has no Vivado. Here is what *was* checked:
> * all Tcl runs under `tclsh` (Tcl 9.0) with stub Vivado/xsdb commands: argument
>   handling, error paths, the order of commands, and the BD connections and address
>   that get requested;
> * `cpl_top_bd.v` is lint-clean (`verilator -Wall`, yosys `hierarchy -check`)
>   against a stub `cpl_top` that has the agreed ports;
> * the smoke test's expected values agree with an independent Python model *and*
>   with the host ISA simulator (`fpga/host/cpl/sim.py`), 40/40 checks.
>
> Expect to fix small things on the first real run, most likely the CIPS
> configuration keys (see [CIPS settings](#cips-settings)).

## Files

| file | what it does | run with |
|---|---|---|
| `files.tcl` | RTL file list in compile order (`cpl_pkg` first), shared by the scripts | sourced by the scripts |
| `common.tcl` | option parsing (`-name value`, plus env fallback), part/licence check | sourced by the scripts |
| `synth_ooc.tcl` | out-of-context synthesis of `cpl_top` at 200 MHz, reports. **Run this first** | `vivado -mode batch` |
| `cpl_top_ooc.xdc` | the 5.000 ns clock on `clk_i` and 1 ns port budgets (OOC only) | read by `synth_ooc.tcl` |
| `cpl_top_bd.v` | Verilog-2001 pass-through shim around `cpl_top` with `X_INTERFACE_*` attributes, used as the IP-integrator module reference (IPI does not accept an SV top here) | used by `bd_cips.tcl` |
| `bd_cips.tcl` | block design: CIPS → SmartConnect → `cpl_top_bd`, proc_sys_reset, address map | sourced by `build_versal.tcl` (or the GUI) |
| `build_versal.tcl` | project-mode build → `out/cpl_versal.pdi` and `out/cpl_versal.xsa` | `vivado -mode batch` |
| `program_jtag.tcl` | programs the `.pdi` over JTAG (Hardware Manager, or xsdb `device program`) | `vivado -mode batch` or `xsdb` |
| `xsdb_smoke.tcl` | ID check, then a WLOAD + MATVEC RAW32 + END program, checked word for word | `xsdb` |
| `smoke_matvec_raw32.hex` | the same smoke test as an op file (`xsdb_smoke.tcl -emit` writes it) | `xsdb xsdb_smoke.tcl -hex …` |

All scripts take options after `-tclargs` (Vivado) or after the script name (xsdb).
They reject unknown options. Outputs go to `fpga/vivado/out/` unless you pass `-out`.

## 0. Prerequisites

* **Vivado 2024.2 or 2025.1** (Linux), with **Versal Premium** device support
  installed. The scripts print a warning on any other version.
* **A licence that covers the VP1202.** Versal Premium needs a paid (Enterprise)
  licence; the free Vivado ML Standard edition does not include this device. To
  check, open the Vivado Tcl console and run `get_parts xcvp1202*`. An empty result
  means the device is either not installed or not licensed. *Help → Manage
  License* lists the licences you have. `synth_ooc.tcl` makes the same check and
  stops with a clear message if it fails.
* **Enough RAM and disk.** Versal Premium implementation is memory hungry. Look up
  AMD's "memory recommendations" for the VP1202 in UG973 (Vivado release notes)
  for your version before you start.
* **To program the card:** a JTAG path to the device (see [BoostX caveats](#boostx--mangoboost-caveats)),
  plus `hw_server`, which comes with Vivado and with Vivado Lab Edition.
* **Confirm the real part on the card.** VP1202 is the planning assumption and has
  not been confirmed for this BoostX. Hardware Manager shows the device name once
  JTAG is connected. The package and speed grade are in the vendor documentation.
  Pass the real values with `-part` or `CPL_PART`.

## 1. Out-of-context synthesis (first)

```sh
cd fpga/vivado
vivado -mode batch -notrace -source synth_ooc.tcl -log out/synth_ooc.log -journal out/synth_ooc.jou
# options: -part <part>  -jobs N  -impl 1 (adds OOC place & route)
#          -generic {GBUF_BYTES=65536 PIPE_TREE=1}   (parameter names: see fpga/rtl/cpl_top.sv)
#          -generic {GBUF_RAM_STYLE=ultra}           GBUF in 16 URAM288 instead of 128 RAMB36
```

Defaults after the timing review: `PIPE_TREE=2`, `GBUF_RD_LAT=3`, `GBUF_REQ_REG=1`
(GBUF port-A latency 4); the URAM path uses the NO_CHANGE coding UltraRAM needs. After
a URAM build, check that `report_ram_utilization` shows 16 URAM288 for the GBUF.

You need only the part and the RTL, so this step finds RTL, licence and
inference problems before any block design work. Read `out/ooc_summary.txt` first, then:

* **`ooc_primitives.rpt`**: the latch count must be 0. The CIM array is
  2 banks × 32 × 64 × 8 b = **32 768 flip-flops** by design (`cpl_bitcell`, ARCH.md §2).
* **GBUF (512 KiB = 4 Mbit) must come out as URAM/BRAM.** Check `ooc_util_hier.rpt`
  for the `cpl_gbuf` instance, and `ooc_ram_util.rpt`. If it shows up as LUTRAM or
  flip-flops, its coding style did not infer a RAM. Fix the RTL; do not lower the
  clock. The ACC buffer (BRAM) and the exp LUTs are handled the same way.
* **`ooc_timing_summary.rpt`** at 5.000 ns. Post-synthesis WNS has no routing in it,
  so treat a small positive value as "probably OK" until `-impl 1` or the full
  build confirms it.

## 2. Block design and device image

```sh
vivado -mode batch -notrace -source build_versal.tcl -log out/build_versal.log -journal out/build_versal.jou
# -to bd      stop after the block design (inspect it: vivado out/proj/cpl_versal.xpr)
# -to synth   stop after synthesis
# -pmc_ref_mhz 33.3333   -pl0_mhz 200   -base 0xA4000000   -strategy Performance_Explore
```

What gets built (`bd_cips.tcl`):

```
versal_cips  "cips"   PS on, JTAG debug, no DDR / NoC / CPM
   pl0_ref_clk (200 MHz) ──┬─ m_axi_fpd_aclk, sc/aclk, rst_pl0/slowest_sync_clk, cpl/clk_i
   pl0_resetn ─────────────┴─ rst_pl0/ext_reset_in
   M_AXI_FPD (AXI4 128b) ── axi_smartconnect "sc" (AXI4 → AXI4-Lite 32b) ── cpl/s_axil
proc_sys_reset "rst_pl0":  interconnect_aresetn → sc/aresetn, peripheral_aresetn → cpl/rst_ni
cpl/irq_o (level high, = STATUS.done | error) → cips/pl_ps_irq0 if this CIPS version
accepts PS_IRQ_USAGE (bd_cips.tcl warns and leaves it open otherwise; polling STATUS
always works). cpl_top uses a synchronous reset, so rst_ni must stay synchronous to
clk_i (peripheral_aresetn is).
cpl_top_bd "cpl" (module reference): s_axil = 2 MiB at 0xA400_0000
```

The cpl window starts at **`0xA400_0000`**, which is the start of the 32-bit M_AXI_FPD
aperture (`0xA400_0000`–`0xAFFF_FFFF`), and is 2 MiB long. So:
`ID` = `0xA400_0000`, IMEM = `0xA401_0000`, GBUF = `0xA410_0000`. The script reads
the assignment back and stops if it is wrong. If you change `-base`, use the same
value in `xsdb_smoke.tcl -base` and in the host driver.

Outputs: `out/cpl_versal.pdi`, `out/cpl_versal.xsa` (written with `write_hw_platform
-fixed -include_bit`, for Vitis, PetaLinux or Yocto), `out/impl_timing_summary.rpt`,
`out/impl_util_hier.rpt` and `out/build_summary.txt`. A Versal device image is made
by `launch_runs impl_1 -to_step write_device_image`; there is no `write_bitstream`.
**If `build_summary.txt` says TIMING FAIL, do not program that image.** Lower
`-pl0_mhz` or fix the paths first.

### CIPS settings

The CIPS configuration is board-specific, and its Tcl key names are version-dependent.
The values that depend on the board are at the top of `bd_cips.tcl`:

| variable | default | where to find the BoostX value |
|---|---|---|
| `CPL_PMC_REF_CLK_MHZ` (`-pmc_ref_mhz`) | 33.3333 (AMD eval boards) | oscillator on the PMC `REF_CLK` pin: board schematic, or `PMC_REF_CLK_FREQMHZ` in the vendor's Vivado project |
| `CPL_DEBUG_MODE` | JTAG | keep JTAG for development |
| `CPL_PMC_EXTRA` | empty | extra `PS_PMC_CONFIG` pairs (UART / MIO for Linux later); copy them from the vendor project |
| `CPL_BASE_ADDR` (`-base`) | 0xA4000000 | free choice inside the M_AXI_FPD aperture |

The script sets `CONFIG.PS_PMC_CONFIG` using the key names that Vivado 2022.1 and
later write in exported BD scripts: `DESIGN_MODE 1`, `PS_USE_M_AXI_FPD 1`,
`PS_USE_PMCPL_CLK0 1`, `PMC_CRP_PL0_REF_CTRL_FREQMHZ 200`, `PS_NUM_FABRIC_RESETS 1`,
`PMC_REF_CLK_FREQMHZ`, and `*_MODE Custom`. If your version rejects the dictionary,
or a pin such as `pl0_ref_clk` or `M_AXI_FPD` is missing, the script stops and names
the pin. To fix it by hand: run `-to bd`, open the project, and double-click CIPS.
Then set *PS PMC → PL interfaces*: M_AXI_FPD on; *Clocking → Output clocks*: PL CLK 0
= 200 MHz; one PL fabric reset; no DDR or NoC. Finally run `write_bd_tcl` and copy the
key names it prints into `bd_cips.tcl`.

## 3. Program over JTAG

Read [BoostX caveats](#boostx--mangoboost-caveats) first.

```sh
hw_server &                                   # on the machine the JTAG cable is attached to
vivado -mode batch -notrace -source program_jtag.tcl -tclargs -pdi out/cpl_versal.pdi
# or, without Vivado (Vitis / Lab Edition):
xsdb program_jtag.tcl -pdi out/cpl_versal.pdi
# options: -url host:3121   -cable <substring of the cable name>   -device 'xcvp1202*'
```

Vivado programs the first `hw_device` that matches `xcvp1202*`, so it skips any
`arm_dap_*` entry. xsdb selects the top-level `Versal*` target and runs `device program`.

## 4. Smoke test

```sh
xsdb xsdb_smoke.tcl                     # built-in test, base 0xA4000000
xsdb xsdb_smoke.tcl -target '*A72*#0'   # force one target
xsdb xsdb_smoke.tcl -hex my_ops.hex     # your own op file (format: header of xsdb_smoke.tcl)
```

1. Reads `ID` and expects `0x43504C31` ("CPL1"). If that fails, the script stops
   without touching anything else.
2. Runs CLEAR and checks STATUS.
3. Writes W (32 × 64 INT8), x (64 INT8) and a poisoned output area into GBUF, and
   reads one word back.
4. Writes a WLOAD, MATVEC (RAW32, 32 columns) and END program into IMEM, and reads
   one word back.
5. Sets START, polls STATUS until done, then checks `PC == 2`, `INSTRET == 3` and
   `PASSES == 1`.
6. Compares the 32 RAW32 outputs with the values computed in Tcl. The script prints
   `SMOKE PASS` or `SMOKE FAIL`.

**Which target to use:** xsdb `mwr`/`mrd` go through a processor's view of the
address map. `Cortex-A72 #0` comes first. After a JTAG-only PDI that core may be
held in reset; if so, the script tries `stop`, then `rst -processor`. `Cortex-R5 #0`
and `PMC` are fallbacks, and it is not verified that they can reach M_AXI_FPD. Run
`targets` in xsdb to see the real names. **Never read the window before the PDI is
loaded.** An access to an aperture where no slave responds can hang the PS
interconnect, and only a power cycle recovers it. SLVERR (an unaligned or unmapped
address, or IMEM/GBUF while busy) shows up as an xsdb memory error and is harmless.

## 5. Host driver

`fpga/host/cpl/driver.py`:

```python
from cpl.driver import Device, XsdbBackend
dev = Device(XsdbBackend(0xA4000000, url="tcp:localhost:3121", target="*A72*#0"))
# url may also be "localhost:3121" (the form xsdb_smoke.tcl / program_jtag.tcl take)
dev.check_id(); print(dev.cfg())
```

Use the same target pattern that worked for `xsdb_smoke.tcl`. Note that XsdbBackend
does not reset or halt the core itself; if the A72 needed `rst -processor`, pass it
through `init=[...]`. JTAG is slow, since every access is a debugger round trip. For real
workloads, run Linux on the APU (PetaLinux or Yocto from `cpl_versal.xsa`, which needs
DDR, and therefore a NoC/DDR configuration from the vendor's board files) and use
`DevMemBackend(0xA4000000)`.

## BoostX / MangoBoost caveats

* **The card normally runs MangoBoost's own shell.** Programming `cpl_versal.pdi`
  replaces the whole device configuration, including whatever drives the card's
  PCIe (CPM) link.
* **The host can hang or crash.** When the link drops, the host sees a surprise
  link-down. AER/DPC handling, a machine check or a hung driver can follow. Do this
  on a machine where a crash is acceptable. Unbind the vendor driver and remove the
  device first (`echo 1 > /sys/bus/pci/devices/<BDF>/remove`). Expect to reboot to
  get the vendor shell back.
* **Know how to restore the vendor image before you start.** It is normally in the
  card's flash (OSPI/QSPI/eMMC), and a power cycle reloads it, because a JTAG-loaded
  PDI does not persist. Never write the flash with these scripts. Ask MangoBoost for
  the recovery procedure and image.
* **Check that JTAG is reachable at all.** Look for a JTAG header, a USB-JTAG
  bridge, or JTAG through the card's BMC or management controller. Some data-center
  cards do not expose it. Without JTAG, steps 3 and 4 cannot be done as written, and
  you need the vendor's own tooling to load a user image.
* **Boot mode.** If the card boots from flash, a JTAG download normally still works,
  because Hardware Manager or xsdb resets the PMC and loads the image. Some setups
  need the boot mode switched to JTAG first. Check the vendor documentation, and
  AMD's answer records for "Versal change boot mode to JTAG" for your version.
* **The PL may belong to the vendor.** If MangoBoost exposes only a user or role
  region (a DFX partition) inside its shell, the right flow is their partial
  reconfiguration flow with `cpl_top` as the role, not this full-device image.
  Ask them before you build.

### PCIe host access (out of scope for v1, sketch)

This v1 design is reached only from the on-card PS (M_AXI_FPD) or JTAG. For a PCIe
host to reach the same AXI4-Lite slave, you need one of these:

1. **The vendor shell.** MangoBoost's PCIe/DMA path forwards one BAR window to a
   user AXI4-Lite port, and `cpl_top_bd` sits behind it. This is the least work if
   they support it.
2. **Your own CPM design.** Enable CPM5 in CIPS in PCIe endpoint mode, using QDMA or
   the AXI-bridge functionality (the AXI-bridge flavour is the AXI Bridge for PCIe in
   CPM). Map BAR0 or BAR2 through its AXI master port, optionally through the NoC,
   into a SmartConnect, then `cpl/s_axil`. On the host, `DevMemBackend` would mmap the
   BAR (`/sys/bus/pci/devices/<BDF>/resource2`) instead of `/dev/mem`. This needs the
   card's PCIe lane, refclk and reset pin-out from MangoBoost. It is also exactly the
   part that conflicts with their shell.

## What is untested

Everything that needs Vivado, xsdb or the card:

* that Vivado 2024.2/2025.1 accepts every option exactly as written (they were
  checked against the documentation only);
* the CIPS `PS_PMC_CONFIG` keys and the top-level `CONFIG.*_MODE` keys;
* module-reference inference of `s_axil`, `clk_i` and `rst_ni` from the attributes;
* SmartConnect conversion from AXI4 128b to AXI4-Lite 32b;
* the address segment name `cpl/s_axil/reg0` (a fallback lookup is in place);
* the name of the PDI file in `impl_1`;
* OOC place and route (`-impl 1`) without `HD.CLK_SRC`;
* the Hardware Manager device name and the xsdb target names on the BoostX;
* whether the A72, R5 or PMC targets can access M_AXI_FPD after a JTAG-only boot;
* `rst -processor -stop` (it falls back to `rst -processor`).
