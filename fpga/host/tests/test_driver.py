"""Driver: Device over SimBackend, DevMemBackend (on a plain file) and XsdbBackend (FakeXsdb)."""
import os
import sys

import numpy as np
import pytest

from cimple_rtl.model import cimple_model as gm
from cpl import compiler as C
from cpl import isa
from cpl.driver import Device, DevMemBackend, SimBackend, XsdbBackend, XsdbError, hw_server_url
from cpl.sim import CplSim, SlvErr

FAKE = os.path.join(os.path.dirname(__file__), "fake_xsdb.py")


def _enc_build(N=48, dm=64, d=64, seed=0, causal=False):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, dm))
    Ws = [rng.standard_normal((dm, d)) / np.sqrt(dm) for _ in range(3)]
    cal = C.calibrate(X, *Ws, causal=causal)
    return C.encoder_attention_head(cal.xq, cal.wq, cal.wk, cal.wv, cal.scales, causal=causal)


def _exercise(dev: Device, small: bool = False):
    dev.check_id()
    cfg = dev.cfg()
    assert cfg["imem_depth"] == 256
    # unaligned byte writes / reads (read-modify-write at the edges)
    dev.write_gbuf(0x100, bytes(range(16)))
    dev.write_gbuf(0x103, b"\xAA\xBB\xCC\xDD\xEE")
    assert dev.read_gbuf(0x100, 16) == bytes([0, 1, 2, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE]) + bytes(range(8, 16))
    assert dev.read_gbuf(0x105, 3) == b"\xCC\xDD\xEE"
    # LUTLOAD through the 2-slot stub
    lut = gm.build_exp_lut(0.03)
    st = dev.load_lut(lut, 0x400)
    assert st["done"] and st["pc"] == 255 and st["instret"] == 2
    # a full encoder head
    b = _enc_build(N=32 if small else 48)
    outs, tot = dev.execute(b)
    for k in b.layout.outputs:
        assert np.array_equal(outs[k], b.ref[k]), k
    assert tot["passes"] == sum(i.ntok for i in b.program if i.opcode == isa.Op.MATVEC)
    assert tot["instret"] == len(b.program) + tot["runs"]
    # error status
    dev.load_program([isa.nop(), isa.matvec(ntok=1, aaddr=3), isa.end()])
    st = dev.run()
    assert st["error"] and st["err_code"] == isa.Err.ALIGN and st["pc"] == 1 and not st["done"]
    dev.clear()
    assert not dev.status()["error"]


def test_device_sim_backend():
    be = SimBackend()
    dev = Device(be)
    _exercise(dev)


def test_sim_backend_busy_and_timeout():
    sim = CplSim(gbuf_bytes=64 * 1024, autorun=False)
    be = SimBackend(sim)
    dev = Device(be)
    dev.load_program([isa.end()])
    with pytest.raises(TimeoutError):
        dev.run(timeout=0.05)            # nobody clocks the sim
    assert dev.status()["busy"]
    with pytest.raises(RuntimeError):
        dev.run()
    with pytest.raises(SlvErr):
        dev.write_gbuf(0, b"1234")        # GBUF while busy -> SLVERR
    with pytest.raises(SlvErr):
        be.read32(0x24)
    sim.finish()
    assert dev.status()["done"]


def test_devmem_backend_on_file(tmp_path):
    f = tmp_path / "mem.bin"
    f.write_bytes(bytes(1 << 16))
    be = DevMemBackend(base_addr=0, size=1 << 16, path=str(f))
    be.write32(0x10, 0xDEADBEEF)
    assert be.read32(0x10) == 0xDEADBEEF
    be.write_block(0x100, np.arange(10, dtype=np.uint32))
    assert np.array_equal(be.read_block(0x100, 10), np.arange(10))
    be.close()
    assert f.read_bytes()[0x10:0x14] == bytes.fromhex("EFBEADDE")
    with pytest.raises(ValueError):
        DevMemBackend(base_addr=0x123, path=str(f))


@pytest.fixture
def xsdb():
    base = 0xA4000000
    be = XsdbBackend(base, cmd=[sys.executable, FAKE, hex(base), "128"], timeout=20)
    yield be
    be.close()


def test_xsdb_backend(xsdb):
    dev = Device(xsdb)
    _exercise(dev, small=True)
    # SLVERR / bus errors surface as SlvErr, other Tcl errors as XsdbError
    with pytest.raises(SlvErr):
        xsdb.read32(0x24)
    with pytest.raises(SlvErr):
        xsdb.write32(0x100002, 1)                # unaligned
    with pytest.raises(XsdbError):
        xsdb.tcl("bogus 1 2")
    assert xsdb.tcl("connect") == ""
    # bulk transfers larger than one command chunk
    w = np.arange(700, dtype=np.uint32) * 0x01010101
    xsdb.write_block(0x100000, w)
    assert np.array_equal(xsdb.read_block(0x100000, 700), w)


def test_run_demo(capsys):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import run_demo
    assert run_demo.main(["--n", "40", "--d", "32", "--causal"]) == 0
    out = capsys.readouterr().out
    assert "O   vs integer reference: exact" in out and "cosine" in out


def test_hw_server_url_forms():
    """xsdb_smoke.tcl / program_jtag.tcl take host:port, xsdb wants tcp:host:port:
    the same string must work for both (review finding F7)."""
    assert hw_server_url(None) is None
    assert hw_server_url("localhost:3121") == "tcp:localhost:3121"
    assert hw_server_url("10.0.0.5:3121") == "tcp:10.0.0.5:3121"
    assert hw_server_url("tcp:localhost:3121") == "tcp:localhost:3121"
    assert hw_server_url("boardhost") == "tcp:boardhost"


@pytest.mark.parametrize("url", ["localhost:3121", "tcp:localhost:3121"])
def test_xsdb_connect_url(url):
    base = 0xA4000000
    be = XsdbBackend(base, url=url, cmd=[sys.executable, FAKE, hex(base), "128"], timeout=20)
    try:
        assert be.init[0] == "connect -url tcp:localhost:3121"
    finally:
        be.close()
