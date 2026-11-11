//! Read-only self process measurements; independent of allocator counters.
#![allow(dead_code)]
#[derive(Debug)]
pub struct Sample {
    pub entries: u64,
    pub mappings: u64,
    pub resident: u64,
    pub pss: Option<u64>,
    pub rss: u64,
    pub provider_mappings: u64,
    pub provider_resident: u64,
    pub provider_virtual: u64,
}
#[cfg(target_os = "macos")]
pub fn snapshot(label: &str, ranges: &[(usize, u32)]) -> Sample {
    // mach/vm_region.h: VM_REGION_SUBMAP_INFO_COUNT_64 (original version).
    #[repr(C, packed(4))]
    #[derive(Default)]
    struct Region {
        protection: i32,
        max_protection: i32,
        inheritance: u32,
        offset: u64,
        user_tag: u32,
        resident: u32,
        private: u32,
        swapped: u32,
        dirtied: u32,
        refs: u32,
        shadow_depth: u16,
        external_pager: u8,
        share_mode: u8,
        is_submap: u32,
        behavior: i32,
        object_id: u32,
        wired: u16,
        reusable: u32,
    }
    unsafe extern "C" {
        fn mach_vm_region_recurse(
            task: u32,
            address: *mut u64,
            size: *mut u64,
            depth: *mut u32,
            info: *mut i32,
            count: *mut u32,
        ) -> i32;
    }
    let mut sorted = ranges.to_vec();
    sorted.sort_unstable();
    let (mut addr, mut depth) = (0u64, 0u32);
    let (mut entries, mut code_entries, mut code_virtual, mut code_resident) =
        (0u64, 0u64, 0u64, 0u64);
    let (mut provider_mappings, mut provider_resident, mut provider_virtual) = (0, 0, 0);
    // SAFETY: queries our own task with correctly sized output buffers. Region
    // addresses are used only as integers; no mapped memory is dereferenced.
    #[allow(deprecated)]
    unsafe {
        let page = libc::sysconf(libc::_SC_PAGESIZE) as u64;
        loop {
            let mut size = 0;
            let mut info = Region::default();
            let mut count = (std::mem::size_of::<Region>() / 4) as u32;
            let kr = mach_vm_region_recurse(
                libc::mach_task_self(),
                &mut addr,
                &mut size,
                &mut depth,
                (&mut info as *mut Region).cast(),
                &mut count,
            );
            if kr == 1 {
                break;
            } // KERN_INVALID_ADDRESS: end of address space.
            assert_eq!(kr, 0, "mach_vm_region_recurse");
            if info.is_submap != 0 {
                depth += 1;
                continue;
            }
            let i = sorted.partition_point(|r| r.0 < addr as usize);
            let contains = sorted.get(i).is_some_and(|r| (r.0 as u64) < addr + size);
            entries += 1;
            if contains {
                code_entries += 1;
                code_virtual += size;
                code_resident += u64::from(info.resident) * page;
            }
            if info.user_tag == libc::VM_MEMORY_APPLICATION_SPECIFIC_16 as u32 {
                provider_mappings += 1;
                provider_resident += u64::from(info.resident) * page;
                provider_virtual += size;
            }
            addr += size;
        }
        let mut info: libc::mach_task_basic_info = std::mem::zeroed();
        let mut count = libc::MACH_TASK_BASIC_INFO_COUNT;
        assert_eq!(
            libc::task_info(
                libc::mach_task_self(),
                libc::MACH_TASK_BASIC_INFO,
                (&mut info as *mut libc::mach_task_basic_info).cast(),
                &mut count
            ),
            0
        );
        let rss = info.resident_size;
        println!("snapshot={label} page={page} entries={entries} code_entries={code_entries} code_virtual={code_virtual} code_resident={code_resident} rss={rss}");
        Sample {
            entries,
            mappings: code_entries,
            resident: code_resident,
            pss: None,
            rss,
            provider_mappings,
            provider_resident,
            provider_virtual,
        }
    }
}

#[cfg(target_os = "linux")]
pub fn snapshot(label: &str, ranges: &[(usize, u32)]) -> Sample {
    let mut sorted = ranges.to_vec();
    sorted.sort_unstable();
    let maps = std::fs::read_to_string("/proc/self/smaps").unwrap();
    // Identify the shared backing objects from the RX ranges, then include
    // their RW aliases as well. A fresh RX mapping may have zero RSS until it
    // executes: measuring just RX would miss all resident writer pages.
    let object = |line: &str| {
        let fields: Vec<_> = line.split_whitespace().collect();
        (fields.len() >= 6 && fields[4] != "0")
            .then(|| (fields[3].to_owned(), fields[4].to_owned()))
    };
    let header_range = |line: &str| -> Option<(usize, usize)> {
        let (a, b) = line.split_whitespace().next()?.split_once('-')?;
        Some((
            usize::from_str_radix(a, 16).ok()?,
            usize::from_str_radix(b, 16).ok()?,
        ))
    };
    let contains = |start: usize, end: usize| {
        let i = sorted.partition_point(|r| r.0 < start);
        sorted.get(i).is_some_and(|r| r.0 < end)
    };
    let objects: std::collections::BTreeSet<_> = maps
        .lines()
        .filter_map(|line| {
            let (start, end) = header_range(line)?;
            contains(start, end).then(|| object(line)).flatten()
        })
        .collect();
    let proc_maps = std::fs::read_to_string("/proc/self/maps").unwrap();
    let mapping_count = proc_maps
        .lines()
        .filter(|line| {
            header_range(line).is_some_and(|(a, b)| contains(a, b))
                || object(line).is_some_and(|id| objects.contains(&id))
        })
        .count() as u64;
    let (mut entries, mut code_entries, mut code_virtual, mut code_resident, mut rss) =
        (0u64, 0u64, 0u64, 0u64, 0u64);
    let mut code = false;
    let mut code_pss = 0;
    let (mut provider_mappings, mut provider_resident, mut provider_virtual) = (0, 0, 0);
    let mut provider = false;
    for line in maps.lines() {
        let first = line.split_whitespace().next().unwrap_or("");
        if let Some((start, end)) = first.split_once('-') {
            let (start, end) = (
                usize::from_str_radix(start, 16).unwrap(),
                usize::from_str_radix(end, 16).unwrap(),
            );
            code = contains(start, end) || object(line).is_some_and(|id| objects.contains(&id));
            let fields: Vec<_> = line.split_whitespace().collect();
            // Include released executable anonymous batches and named backing
            // views as well as current units, so stale mappings cannot disappear
            // from the witness just because provider accounting forgot them.
            provider = code
                || line.contains("/memfd:x86jit-code")
                || (fields[1].contains('x')
                    && (fields.len() == 5 || line.ends_with("/dev/zero (deleted)")));
            if provider {
                provider_mappings += 1;
                provider_virtual += (end - start) as u64;
            }
            entries += 1;
            if code {
                code_entries += 1;
                code_virtual += (end - start) as u64;
            }
        } else if first == "Pss:" && code {
            code_pss += line
                .split_whitespace()
                .nth(1)
                .unwrap()
                .parse::<u64>()
                .unwrap()
                * 1024;
        } else if first == "Rss:" {
            let resident: u64 = line
                .split_whitespace()
                .nth(1)
                .unwrap()
                .parse::<u64>()
                .unwrap()
                * 1024;
            rss += resident;
            if code {
                code_resident += resident;
            }
            if provider {
                provider_resident += resident;
            }
        }
    }
    assert_eq!(code_entries, mapping_count, "maps/smaps code-view count");
    println!("code_pss={code_pss}");
    // SAFETY: sysconf has no pointer arguments.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as u64 };
    println!("snapshot={label} page={page} entries={entries} code_entries={code_entries} code_virtual={code_virtual} code_resident={code_resident} rss={rss}");
    Sample {
        entries,
        mappings: code_entries,
        resident: code_resident,
        pss: Some(code_pss),
        rss,
        provider_mappings,
        provider_resident,
        provider_virtual,
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Resources {
    pub mappings: u64,
    pub mapped_bytes: u64,
    pub descriptors: usize,
}

#[cfg(any(target_os = "linux", target_os = "freebsd"))]
std::thread_local! {
    static MAPPING_RANGES: std::cell::RefCell<Option<Vec<std::ops::Range<usize>>>> = const {
        std::cell::RefCell::new(None)
    };
}

pub struct MappingWatch;

impl MappingWatch {
    pub fn new() -> Self {
        #[cfg(any(target_os = "linux", target_os = "freebsd"))]
        MAPPING_RANGES.with(|ranges| {
            assert!(ranges.borrow_mut().replace(Vec::new()).is_none());
        });
        Self
    }
}

impl Drop for MappingWatch {
    fn drop(&mut self) {
        #[cfg(any(target_os = "linux", target_os = "freebsd"))]
        MAPPING_RANGES.with(|ranges| *ranges.borrow_mut() = None);
    }
}

#[cfg(any(target_os = "linux", target_os = "freebsd"))]
pub fn observe_mapping(start: usize, len: usize) {
    MAPPING_RANGES.with(|ranges| {
        if let Some(ranges) = ranges.borrow_mut().as_mut() {
            // Keep the union of successful OS reservations, including attempts
            // that fail before publication. Unmapping never edits this evidence.
            let mut range = start..start + len;
            ranges.retain(|old| {
                if old.end < range.start || range.end < old.start {
                    true
                } else {
                    range.start = range.start.min(old.start);
                    range.end = range.end.max(old.end);
                    false
                }
            });
            ranges.push(range);
            ranges.sort_unstable_by_key(|r| r.start);
        }
    });
}

#[cfg(any(target_os = "linux", target_os = "freebsd"))]
fn watched_bytes(start: usize, end: usize) -> u64 {
    MAPPING_RANGES.with(|ranges| {
        ranges
            .borrow()
            .as_ref()
            .expect("MappingWatch must precede resource sampling")
            .iter()
            .map(|r| end.min(r.end).saturating_sub(start.max(r.start)) as u64)
            .sum()
    })
}

pub fn resources() -> Resources {
    #[cfg(target_os = "linux")]
    {
        let mut mappings = 0;
        let mut mapped_bytes = 0;
        for line in std::fs::read_to_string("/proc/self/maps").unwrap().lines() {
            let (start, end) = line
                .split_whitespace()
                .next()
                .unwrap()
                .split_once('-')
                .unwrap();
            let bytes = watched_bytes(
                usize::from_str_radix(start, 16).unwrap(),
                usize::from_str_radix(end, 16).unwrap(),
            );
            mappings += u64::from(bytes != 0);
            mapped_bytes += bytes;
        }
        Resources {
            mappings,
            mapped_bytes,
            descriptors: std::fs::read_dir("/proc/self/fd").unwrap().count(),
        }
    }
    #[cfg(any(target_os = "macos", target_os = "freebsd"))]
    {
        // SAFETY: fcntl only probes descriptor validity; no descriptor is modified.
        let descriptors = unsafe {
            (0..libc::getdtablesize())
                .filter(|&fd| libc::fcntl(fd, libc::F_GETFD) != -1)
                .count()
        };
        #[cfg(target_os = "macos")]
        let (mappings, mapped_bytes) = {
            let sample = snapshot("resource-cleanup", &[]);
            (sample.provider_mappings, sample.provider_virtual)
        };
        #[cfg(target_os = "freebsd")]
        let (mappings, mapped_bytes) = {
            let mut count = 0;
            let (mut mappings, mut bytes) = (0, 0);
            // SAFETY: libutil allocates count entries for this process, then
            // the returned buffer is freed with the documented libc allocator.
            unsafe {
                let entries = libc::kinfo_getvmmap(libc::getpid(), &mut count);
                assert!(!entries.is_null(), "kinfo_getvmmap");
                for entry in std::slice::from_raw_parts(entries, count as usize) {
                    let owned = watched_bytes(entry.kve_start as usize, entry.kve_end as usize);
                    mappings += u64::from(owned != 0);
                    bytes += owned;
                }
                libc::free(entries.cast());
            }
            (mappings, bytes)
        };
        Resources {
            mappings,
            mapped_bytes,
            descriptors,
        }
    }
}
