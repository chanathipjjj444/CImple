"""Stand-in for Vivado's xsdb, speaking the subset XsdbBackend uses.

Usage: python fake_xsdb.py <base_addr> [gbuf_kib]
Reads Tcl lines of the form
    puts "@@BEGIN"; set __rc [catch {<cmd>} __r]; puts "@@RC $__rc"; puts $__r; puts "@@END"
with <cmd> one of: connect ..., targets ..., mwr -force <addr> {<w> ...},
mrd -force <addr> <n>; addresses outside [base, base + 2 MiB) or answered
with SLVERR by the simulated device give a Tcl error, like a bus error on
real hardware. Prints an "xsdb% " prompt before each reply, as the real
interactive shell does. Backed by cpl.sim.CplSim (autorun).
"""
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cpl.sim import OKAY, CplSim  # noqa: E402

WRAP = re.compile(r'^puts "@@BEGIN"; set __rc \[catch \{(.*)\} __r\]; puts "@@RC \$__rc"; '
                  r'puts \$__r; puts "@@END"$')


def main():
    base = int(sys.argv[1], 0)
    kib = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    sim = CplSim(gbuf_bytes=kib * 1024)

    def off(a):
        o = a - base
        if not 0 <= o < (1 << 21):
            raise RuntimeError(f"Memory access error at 0x{a:X}: no device")
        return o

    def run(cmd):
        argv = cmd.split(None, 1)
        if not argv or argv[0] in ("connect", "targets"):
            return ""
        if argv[0] == "mwr":
            m = re.fullmatch(r"mwr -force (0x[0-9A-Fa-f]+) \{([^}]*)\}", cmd)
            a = int(m.group(1), 16)
            for i, v in enumerate(m.group(2).split()):
                if sim.axi_write(off(a + 4 * i), int(v, 0)) != OKAY:
                    raise RuntimeError(f"Memory write error at 0x{a + 4 * i:X}. Blocked address")
            return ""
        if argv[0] == "mrd":
            m = re.fullmatch(r"mrd -force (0x[0-9A-Fa-f]+) (\d+)", cmd)
            a, n = int(m.group(1), 16), int(m.group(2))
            lines = []
            for i in range(n):
                d, r = sim.axi_read(off(a + 4 * i))
                if r != OKAY:
                    raise RuntimeError(f"Memory read error at 0x{a + 4 * i:X}. Blocked address")
                lines.append(f"{a + 4 * i:8X}:   {d:08X}")
            return "\n".join(lines)
        raise RuntimeError(f'invalid command name "{argv[0]}"')

    out = sys.stdout
    out.write("xsdb% ")
    out.flush()
    for line in sys.stdin:
        line = line.rstrip("\n")
        if line.strip() == "exit":
            break
        m = WRAP.match(line)
        if m is None:
            out.write(f'invalid command name "{line}"\nxsdb% ')
            out.flush()
            continue
        try:
            res, rc = run(m.group(1)), 0
        except Exception as e:  # Tcl error
            res, rc = str(e), 1
        out.write(f"@@BEGIN\n@@RC {rc}\n{res}\n@@END\nxsdb% ")
        out.flush()


if __name__ == "__main__":
    main()
