#!/usr/bin/env python3
"""Synthesis-structure regression checks for fpga/rtl (yosys, no Vivado).

    python3 check_synth.py <rtl_dir> <work_dir>
    (make -C fpga/tb synth-check  [SV2V=... YOSYS="..."])

Environment: SV2V (default "sv2v") and YOSYS (default "yowasp-yosys"; may be
a command line such as "uv tool run --from yowasp-yosys yowasp-yosys").

Checks (review findings, fpga/docs/ARCH.md is silent on all of them):

  uram   cpl_gbuf with RAM_STYLE="ultra" maps onto URAM288 with
         `synth_xilinx -family xcup` (the yosys URAM288 model, like the
         primitive, only has NO_CHANGE ports: a read-first port A fails
         with "no valid mapping found for memory"). 64 KiB -> 2 URAM288.
  areset no flip-flop anywhere in cpl_top has an asynchronous reset
         (rst_ni comes from proc_sys_reset, synchronous to clk_i).
  rstce  no flip-flop without a reset has rst_ni in the cone of its D / EN
         inputs, i.e. no non-reset register sits in a reset process (rst_ni
         would become part of its clock enable: win_q / wmask_q /
         gb_wdata_q in cpl_engine before the fix).
  noreset the datapath registers that must not be reset: cpl_partition
         acc_q, cpl_engine ins_q / win_q / wmask_q / gb_wdata_q / wtok.

A self-test first runs `rstce` / `areset` on two tiny modules that violate
them, so a check that silently finds nothing fails loudly instead.
Exit status 0 = all PASS.
"""
import json
import os
import shlex
import subprocess
import sys
from collections import defaultdict

SRCS = ["cpl_pkg.sv", "cpl_bitcell.sv", "cpl_oai221.sv", "cpl_macro.sv", "cpl_partition.sv",
        "cpl_core.sv", "cpl_quant.sv", "cpl_lut.sv", "cpl_post.sv", "cpl_gbuf.sv",
        "cpl_imem.sv", "cpl_engine.sv", "cpl_axil.sv", "cpl_top.sv"]

SEQ_PREFIX = ("$dff", "$dffe", "$sdff", "$adff", "$aldff", "$dffsr", "$sr", "$dlatch",
              "$adlatch", "$mem", "$memrd", "$memwr", "$meminit", "$ff")
ASYNC_TYPES = ("$adff", "$adffe", "$aldff", "$aldffe", "$dffsr", "$dffsre", "$adlatch")
SYNC_RESET_TYPES = ("$sdff", "$sdffe", "$sdffce")

SV2V = shlex.split(os.environ.get("SV2V", "sv2v"))
YOSYS = shlex.split(os.environ.get("YOSYS", "yowasp-yosys"))


def run(cmd, log):
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return r.returncode


def sv2v(files, out):
    with open(out, "w") as f:
        r = subprocess.run(SV2V + files, stdout=f, stderr=subprocess.PIPE, text=True)
    if r.returncode:
        raise RuntimeError(f"sv2v failed: {r.stderr}")


def yosys(script, log):
    return run(YOSYS + ["-q", "-p", script], log)


# ---------------------------------------------------------------------------
# netlist cone analysis on yosys write_json output (not flattened)
# ---------------------------------------------------------------------------
def is_seq(ctype):
    # flip-flops / memories, and submodule instances (plain or "$paramod..." names)
    return (ctype.startswith(SEQ_PREFIX) or not ctype.startswith("$")
            or ctype.startswith("$paramod"))


def analyse(json_path, mod_filter=None):
    """Return (async_ffs, folded, ffs_by_name) per module."""
    with open(json_path) as f:
        nl = json.load(f)
    res = {}
    for mname, mod in nl["modules"].items():
        if mod_filter and not any(k in mname for k in mod_filter):
            continue
        rst_bits = set()
        for pname, p in mod.get("ports", {}).items():
            if pname == "rst_ni":
                rst_bits |= {b for b in p["bits"] if isinstance(b, int)}
        driver = {}
        for cname, c in mod["cells"].items():
            dirs = c.get("port_directions", {})
            for pn, bits in c["connections"].items():
                if dirs.get(pn) == "output":
                    for b in bits:
                        if isinstance(b, int):
                            driver[b] = cname
        names = defaultdict(set)                     # bit -> net names
        for nn, n in mod["netnames"].items():
            for b in n["bits"]:
                if isinstance(b, int):
                    names[b].add(nn)

        def cone_has_rst(bits):
            seen, todo = set(), [b for b in bits if isinstance(b, int)]
            while todo:
                b = todo.pop()
                if b in seen:
                    continue
                seen.add(b)
                if b in rst_bits:
                    return True
                cn = driver.get(b)
                if cn is None:
                    continue
                c = mod["cells"][cn]
                if is_seq(c["type"]):
                    continue
                dirs = c.get("port_directions", {})
                for pn, pb in c["connections"].items():
                    if dirs.get(pn) == "input":
                        todo.extend(x for x in pb if isinstance(x, int))
            return False

        async_ffs, folded, ffs = [], [], {}
        for cname, c in mod["cells"].items():
            t = c["type"]
            if (not t.startswith(SEQ_PREFIX) or t.startswith("$mem")
                    or t.startswith("$paramod")):
                continue
            q = c["connections"].get("Q", [])
            qnames = sorted({n for b in q if isinstance(b, int) for n in names[b]
                             if not n.startswith("$")}) or [cname]
            ffs[cname] = (t, qnames)
            if t.startswith(ASYNC_TYPES):
                async_ffs.append((t, qnames))
            if not t.startswith(SYNC_RESET_TYPES) and not t.startswith(ASYNC_TYPES):
                ins = c["connections"].get("D", []) + c["connections"].get("EN", [])
                if rst_bits and cone_has_rst(ins):
                    folded.append((t, qnames))
        res[mname] = (async_ffs, folded, ffs)
    return res


def reset_style(json_path):
    """-> (async list, folded list) over all modules: [(module, type, names)]"""
    a, f = [], []
    for m, (async_ffs, folded, _) in analyse(json_path).items():
        a += [(m, t, n) for t, n in async_ffs]
        f += [(m, t, n) for t, n in folded]
    return a, f


def synth_struct(src_v, top, chparams, out_json, log):
    # -defer: elaborate only the parametrisation hierarchy asks for (the
    # default 512 KiB GBUF would otherwise be elaborated first)
    ch = " ".join(f"-chparam {k} {v}" for k, v in chparams.items())
    script = (f"read_verilog -defer -sv {src_v}; hierarchy -top {top} {ch}; proc; opt_expr; "
              f"opt_clean; opt_dff; opt_clean; write_json {out_json}")
    return yosys(script, log)


# ---------------------------------------------------------------------------
SELFTEST = r"""
module st_bad (input logic clk_i, input logic rst_ni, input logic [7:0] d_i, input logic e_i,
               output logic [7:0] c_o, output logic [7:0] x_o);
  always_ff @(posedge clk_i) begin
    if (!rst_ni) c_o <= '0;
    else begin
      c_o <= c_o + 8'd1;
      if (e_i) x_o <= d_i;     // non-reset register in a reset process
    end
  end
endmodule
module st_async (input logic clk_i, input logic rst_ni, input logic d_i, output logic q_o);
  always_ff @(posedge clk_i or negedge rst_ni)
    if (!rst_ni) q_o <= 1'b0; else q_o <= d_i;
endmodule
"""

# the pre-fix cpl_gbuf "ultra" coding (read-first port A), reduced: must NOT map
URAM_RF = r"""
module uram_rf (input clk, input en, input [15:0] we, input [11:0] aa, input [127:0] wd,
                output reg [127:0] rd, input [15:0] bwe, input [11:0] ba, input [127:0] bwd);
  (* ram_style = "ultra" *) reg [127:0] mem [0:4095];
  reg [127:0] r1; integer i;
  always @(posedge clk) if (en) begin
    for (i = 0; i < 16; i = i + 1) if (we[i]) mem[aa][8*i +: 8] <= wd[8*i +: 8];
    r1 <= mem[aa];
  end
  always @(posedge clk) for (i = 0; i < 16; i = i + 1) if (bwe[i]) mem[ba][8*i +: 8] <= bwd[8*i +: 8];
  always @(posedge clk) rd <= r1;
endmodule
"""

GBUF_WRAP = r"""
module gbuf_uram_wrap import cpl_pkg::*; (
  input  logic clk_i, input logic a_en_i, input logic [15:0] a_we_i,
  input  logic [11:0] a_addr_i, input logic [127:0] a_wdata_i, output logic [127:0] a_rdata_o,
  input  logic [15:0] b_we_i, input logic [11:0] b_addr_i, input logic [127:0] b_wdata_i);
  cpl_gbuf #(.GBUF_BYTES(65536), .RD_LAT(2), .RAM_STYLE("ultra")) u (.*);
endmodule
"""


def main():
    rtl, work = sys.argv[1], sys.argv[2]
    os.makedirs(work, exist_ok=True)
    ok = True

    def report(name, passed, detail=""):
        nonlocal ok
        ok &= passed
        print(f"synth-check {name}: {'PASS' if passed else 'FAIL'}{(' -- ' + detail) if detail else ''}")

    # ---- self-test of the cone analysis --------------------------------------
    st_sv = os.path.join(work, "selftest.sv")
    with open(st_sv, "w") as f:
        f.write(SELFTEST)
    sv2v([st_sv], os.path.join(work, "selftest.v"))
    rc = synth_struct(os.path.join(work, "selftest.v"), "st_bad", {},
                      os.path.join(work, "st_bad.json"), os.path.join(work, "st_bad.log"))
    _, fb = reset_style(os.path.join(work, "st_bad.json")) if rc == 0 else ([], [])
    rc2 = synth_struct(os.path.join(work, "selftest.v"), "st_async", {},
                       os.path.join(work, "st_async.json"), os.path.join(work, "st_async.log"))
    ab, _ = reset_style(os.path.join(work, "st_async.json")) if rc2 == 0 else ([], [])
    report("selftest", rc == 0 and rc2 == 0 and any("x_o" in n for _, _, n in fb)
           and not any("c_o" in n for _, _, n in fb) and len(ab) == 1,
           f"folded={[n for _, _, n in fb]} async={[n for _, _, n in ab]}")

    # ---- uram -----------------------------------------------------------------
    wrap = os.path.join(work, "gbuf_wrap.sv")
    with open(wrap, "w") as f:
        f.write(GBUF_WRAP)
    gv = os.path.join(work, "gbuf_uram.v")
    sv2v([os.path.join(rtl, "cpl_pkg.sv"), os.path.join(rtl, "cpl_gbuf.sv"), wrap], gv)
    stat = os.path.join(work, "gbuf_uram_stat.json")
    # stop before LUT mapping: the memory decision is made in map_memory, and
    # ABC on the remaining fabric logic only costs time
    rc = yosys(f"read_verilog -defer -sv {gv}; synth_xilinx -family xcup -noiopad -top gbuf_uram_wrap "
               f"-run :fine; "
               f"tee -q -o {stat} stat -json", os.path.join(work, "gbuf_uram.log"))
    n_uram = 0
    if rc == 0:
        with open(stat) as f:
            st = json.load(f)
        for m in st.get("modules", {}).values():
            n_uram += int(m.get("num_cells_by_type", {}).get("URAM288", 0))
        n_uram = n_uram or int(st.get("design", {}).get("num_cells_by_type", {}).get("URAM288", 0))
    report("uram", rc == 0 and n_uram == 2,
           f"yosys rc={rc}, URAM288={n_uram} (expect 2; log {os.path.join(work, 'gbuf_uram.log')})")

    # the check must be able to fail: the old read-first coding is rejected
    rf = os.path.join(work, "uram_rf.v")
    with open(rf, "w") as f:
        f.write(URAM_RF)
    rc_rf = yosys(f"read_verilog {rf}; synth_xilinx -family xcup -noiopad -top uram_rf -run :fine",
                  os.path.join(work, "uram_rf.log"))
    report("uram-selftest", rc_rf != 0, f"read-first URAM coding: yosys rc={rc_rf} (expect a "
           "'no valid mapping' error)")

    # ---- reset style over the whole design ---------------------------------------
    tv = os.path.join(work, "cpl_top.v")
    sv2v([os.path.join(rtl, s) for s in SRCS], tv)
    tj = os.path.join(work, "cpl_top.json")
    rc = synth_struct(tv, "cpl_top", {"GBUF_BYTES": 16384}, tj, os.path.join(work, "cpl_top.log"))
    if rc != 0:
        report("areset", False, f"yosys rc={rc}")
        report("rstce", False, f"yosys rc={rc}")
        report("noreset", False, f"yosys rc={rc}")
        return 1
    res = analyse(tj)
    a = [(m, t, n) for m, (af, _, _) in res.items() for t, n in af]
    fo = [(m, t, n) for m, (_, fl, _) in res.items() for t, n in fl]
    report("areset", not a, f"{len(a)} async-reset FF cells" + (f": {a[:6]}" if a else ""))
    report("rstce", not fo, f"{len(fo)} non-reset FF cells with rst_ni in D/EN"
           + (f": {fo[:6]}" if fo else ""))
    want = {"cpl_partition": ["acc_q"],
            "cpl_engine": ["ins_q", "win_q", "wmask_q", "gb_wdata_q", "wtok"]}
    bad, found = [], set()
    for m, (_, _, ffs) in res.items():
        for base, sigs in want.items():
            if base not in m:
                continue
            for t, qn in ffs.values():
                for s in sigs:
                    if any(n == s or n.startswith(s + "[") or n.startswith(s + ".") for n in qn):
                        found.add((base, s))
                        if t.startswith(SYNC_RESET_TYPES) or t.startswith(ASYNC_TYPES):
                            bad.append((m, t, s))
    missing = [(b, s) for b, ss in want.items() for s in ss if (b, s) not in found]
    report("noreset", not bad and not missing, f"reset={bad} not-found={missing}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
