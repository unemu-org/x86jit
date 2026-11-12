use x86jit_core::{
    lift::{lift_block, FetchAddr},
    CpuMode, IrBlock, IrRegion, Memory, MemoryModel, Prot, RegionKind,
};

pub fn memory(code: &[u8]) -> Memory {
    let mut mem = Memory::new(MemoryModel::Flat { size: 0x4000 });
    mem.map(0, 0x4000, Prot::RWX, RegionKind::Ram).unwrap();
    for (i, byte) in code.iter().enumerate() {
        mem.write(0x1000 + i as u64, u64::from(*byte), 1).unwrap();
    }
    mem
}

pub fn block(code: &[u8], mode: CpuMode) -> IrBlock {
    lift_block(&memory(code), FetchAddr::flat(0x1000), mode).unwrap()
}

pub fn blocks(mode: CpuMode) -> Vec<(&'static str, IrBlock, u32)> {
    let wide = mode == CpuMode::Long64;
    vec![
        ("push-pop", block(b"\x50\x5b\xf4", mode), 0),
        ("call", block(b"\xe8\x10\x00\x00\x00", mode), 0),
        ("ret", block(b"\xc3", mode), 0),
        ("linked", block(b"\xeb\xfe", mode), 0),
        (
            "memory",
            block(
                if wide {
                    b"\x48\x8b\x03\x48\x89\x07\xf4"
                } else {
                    b"\x8b\x03\x89\x07\xf4"
                },
                mode,
            ),
            0,
        ),
        (
            "tls",
            block(
                if wide {
                    b"\x64\x48\x8b\x03\x64\x48\x89\x07\xf4"
                } else {
                    b"\x64\x8b\x03\x64\x89\x07\xf4"
                },
                mode,
            ),
            0,
        ),
        ("rep-movs", block(b"\xf3\xa4\xf4", mode), 0),
        ("rep-stos", block(b"\xf3\xaa\xf4", mode), 0),
        ("rep-movs-9fc0", block(b"\xf3\xa4\xf4", mode), 0x8040),
        ("rep-stos-9fc0", block(b"\xf3\xaa\xf4", mode), 0x8040),
        ("vrcpps-9fc0", block(b"\xc5\xf8\x53\xc1\xf4", mode), 0x8040),
        ("mxcsr", block(b"\xf3\x0f\x58\xc1\xf4", mode), 0x2000),
    ]
}

pub fn region(mode: CpuMode) -> IrRegion {
    // A conditional diamond exercises register flow across internal region edges.
    let mem = memory(b"\x85\xc0\x74\x03\x83\xc0\x01\xf4");
    IrRegion {
        entry: 0x1000,
        blocks: [0x1000, 0x1004, 0x1007]
            .into_iter()
            .map(|pc| lift_block(&mem, FetchAddr::flat(pc), mode).unwrap())
            .collect(),
        has_loop: false,
    }
}

pub fn real16() -> Vec<(&'static str, IrBlock)> {
    [
        ("alu", &b"\xb8\x34\x12\x01\xd8\x46\xf4"[..]),
        ("stack", &b"\x50\x5b\xe8\x04\x00"[..]),
        ("memory", &b"\x8b\x00\x89\x06\x00\x20\xf4"[..]),
        ("rep", &b"\xf3\xa5\xf4"[..]),
        ("sizes", &b"\x66\x8b\x03\x67\x8b\x03\xf4"[..]),
        ("flags-all", &b"\x9c\x9d\x60\x61\xf4"[..]),
        ("far-call", &b"\x9a\x00\x20\x00\x01"[..]),
        ("segments", &b"\x8e\xd8\x8e\xd0\x8b\x46\x00\xf4"[..]),
        ("prefix-combination", &b"\x66\x67\x8b\x03\xf4"[..]),
    ]
    .into_iter()
    .map(|(name, code)| (name, block(code, CpuMode::Real16)))
    .collect()
}
