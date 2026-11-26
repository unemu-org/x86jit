//! Integer-key hashing for the shared translation map (§9.1) and the Machine
//! interpreter's per-vCPU block and context maps (decision-23 §5).
use std::hash::Hasher;

// Keys are guest addresses under the embedder's control, or values the engine
// derives from them and from its own state. HashDoS resistance is out of scope:
// whoever can choose a colliding key set already controls the guest CPU the map
// serves, so collisions only slow that guest down. Avoid SipHash's cost on
// every dispatch lookup.
#[derive(Default)]
pub struct BlockHasher(u64);

impl Hasher for BlockHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0 ^ (self.0 >> 32)
    }

    #[inline]
    fn write_u64(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(0x517c_c1b7_2722_0a95);
    }

    // Narrow integers take one round each, not the byte path below.
    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.write_u64(u64::from(i));
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.write_u64(u64::from(i));
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.write_u64(u64::from(i));
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.write_u64(i as u64);
    }

    fn write(&mut self, bytes: &[u8]) {
        let (chunks, tail) = bytes.as_chunks::<8>();
        for chunk in chunks {
            self.write_u64(u64::from_le_bytes(*chunk));
        }
        if !tail.is_empty() {
            let mut word = [0; 8];
            word[..tail.len()].copy_from_slice(tail);
            self.write_u64(u64::from_le_bytes(word));
        }
    }
}

// Avalanche all address bits before selecting a small power-of-two index.
#[inline]
pub(crate) fn dispatch_hash(mut rip: u64) -> u64 {
    rip ^= rip >> 30;
    rip = rip.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    rip ^= rip >> 27;
    rip = rip.wrapping_mul(0x94d0_49bb_1331_11eb);
    rip ^ (rip >> 31)
}
