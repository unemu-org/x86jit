//! NEC V20/V30 ADD4S/SUB4S/CMP4S: the `NecBcdStringOverlap` refusal is exact.
//! The manual does not state the order of overlapping BCD-string transfers, so
//! ADD4S/SUB4S refuse exactly when destination byte i is the source byte of a
//! later step j > i at the same physical address (decision-23 §3). For every
//! placement, including segment aliasing, both offset wraps and the 1 MiB wrap,
//! that predicate must decide the refusal, and every executed case must give the
//! same result under a byte-serial order and an all-reads-first order.
use x86jit_core::{
    CpuModel, Exit, InterpreterBackend, Prot, RegionKind, StepOptions, StepStop,
    UnsupportedSystemOperation, Vm, VmConfig,
};

const IP: u64 = 0x1200;

fn phys(seg: u16, off: u16) -> usize {
    (((u64::from(seg) << 4) + u64::from(off)) & 0xfffff) as usize
}
fn fill(p: usize) -> u8 {
    ((((p * 7 + 3) % 10) << 4) | ((p * 3 + 1) % 10)) as u8
}

// Returns the final memory bytes at the destination addresses and (Z, CY).
fn model(mem: &mut [u8], s: &[usize], d: &[usize], op: u8, reads_first: bool) -> (bool, bool) {
    let n = s.len();
    let mut pairs = Vec::new();
    if reads_first {
        for i in 0..n {
            pairs.push((mem[d[i]], mem[s[i]]));
        }
    }
    let (mut carry, mut zero) = (0i32, true);
    let mut results = Vec::new();
    for i in 0..n {
        let (a, b) = if reads_first {
            pairs[i]
        } else {
            (mem[d[i]], mem[s[i]])
        };
        let mut digit = |a: i32, b: i32| {
            let v = if op == 0x20 {
                a + b + carry
            } else {
                a + 10 - b - carry
            };
            carry = if op == 0x20 {
                (v > 9) as i32
            } else {
                (v < 10) as i32
            };
            v % 10
        };
        let lo = digit(i32::from(a & 15), i32::from(b & 15));
        let hi = digit(i32::from(a >> 4), i32::from(b >> 4));
        let r = ((hi << 4) | lo) as u8;
        zero &= r == 0;
        if op != 0x26 {
            if reads_first {
                results.push(r);
            } else {
                mem[d[i]] = r;
            }
        }
    }
    if reads_first && op != 0x26 {
        for i in 0..n {
            mem[d[i]] = results[i];
        }
    }
    (zero, carry != 0)
}

#[test]
fn bcd_string_overlap_refusal_is_exact() {
    // (DS, IX, ES, IY) placements.
    let mut cases: Vec<(u16, u16, u16, u16)> = Vec::new();
    // Same segment, IY - IX in -8..=8.
    for d in -8i32..=8 {
        cases.push((0x300, 0x40, 0x300, (0x40 + d) as u16));
    }
    // Segment aliasing: ES = DS + k, IY chosen around the same physical bytes.
    for k in -2i32..=2 {
        for d in -4i32..=4 {
            let iy = (0x40 - 16 * k + d) as u16;
            cases.push((0x300, 0x40, (0x300 + k) as u16, iy));
        }
    }
    // Offset wrap inside the segment (IX and/or IY near FFFFh).
    for d in -4i32..=4 {
        cases.push((0x400, 0xfffe, 0x400, (0xfffe_i32 + d) as u16));
        cases.push((0x400, 0xffff, 0x400, (d.rem_euclid(0x10000)) as u16));
    }
    // 1 MiB wrap: ES = FFFFh, IY = 30h + d lands at physical 20h + d.
    for d in -4i32..=4 {
        cases.push((0, 0x20, 0xffff, (0x30 + d) as u16));
        cases.push((0xffff, (0x30 + d) as u16, 0, 0x20));
    }
    let (mut refused, mut ran, mut refused_differs) = (0, 0, 0);
    for &(ds, ix, es, iy) in &cases {
        for n in 1usize..=4 {
            let s: Vec<usize> = (0..n)
                .map(|j| phys(ds, ix.wrapping_add(j as u16)))
                .collect();
            let d: Vec<usize> = (0..n)
                .map(|i| phys(es, iy.wrapping_add(i as u16)))
                .collect();
            let predicate = (1..n).any(|j| (0..j).any(|i| d[i] == s[j]));
            for op in [0x20u8, 0x22, 0x26] {
                let mut base = vec![0u8; 0x100000];
                for &p in s.iter().chain(d.iter()) {
                    base[p] = fill(p);
                }
                let mut serial = base.clone();
                let fs = model(&mut serial, &s, &d, op, false);
                let mut first = base.clone();
                let ff = model(&mut first, &s, &d, op, true);
                let order_shows = serial != first || fs != ff;

                let mut vm =
                    Vm::with_backend(VmConfig::flat(0x120000), Box::new(InterpreterBackend));
                vm.set_cpu_model(CpuModel::NecV20);
                vm.map(0, 0x120000, Prot::RWX, RegionKind::Ram).unwrap();
                for &p in s.iter().chain(d.iter()) {
                    vm.write_bytes(p as u64, &[base[p]]).unwrap();
                }
                vm.write_bytes(IP, &[0x0f, op, 0xf4]).unwrap();
                let mut cpu = vm.new_vcpu();
                cpu.cpu.rip = IP;
                cpu.cpu.gpr[4] = 0x8000;
                cpu.cpu.ds = ds;
                cpu.cpu.es = es;
                cpu.cpu.gpr[6] = u64::from(ix);
                cpu.cpu.gpr[7] = u64::from(iy);
                cpu.cpu.gpr[1] = (2 * n) as u64;
                let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                let tag = format!("op {op:#x} ds {ds:#x} ix {ix:#x} es {es:#x} iy {iy:#x} n {n}");
                let store = op != 0x26;
                if store && predicate {
                    assert!(
                        matches!(
                            r.stop(),
                            StepStop::Exit(Exit::UnsupportedSystemContext {
                                operation: UnsupportedSystemOperation::NecBcdStringOverlap,
                                ..
                            })
                        ),
                        "{tag}: {:?}",
                        r.stop()
                    );
                    for &p in &d {
                        assert_eq!(vm.mem.read(p as u64, 1).unwrap() as u8, base[p], "{tag}");
                    }
                    refused += 1;
                    if order_shows {
                        refused_differs += 1;
                    }
                } else {
                    assert_eq!(r.steps(), 1, "{tag}: {:?}", r.stop());
                    // Executed: the order must not show, and the engine matches it.
                    assert!(!order_shows, "{tag}: executed but order-dependent");
                    for &p in &d {
                        assert_eq!(vm.mem.read(p as u64, 1).unwrap() as u8, serial[p], "{tag}");
                    }
                    assert_eq!((cpu.cpu.flags.zf, cpu.cpu.flags.cf), fs, "{tag}");
                    ran += 1;
                }
            }
        }
    }
    eprintln!("refused {refused} (order visible with this data in {refused_differs}), ran {ran}");
    assert!(refused > 0 && ran > 0);
}
