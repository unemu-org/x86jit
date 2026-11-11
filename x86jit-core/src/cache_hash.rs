//! Integer-key hashing for the shared translation map (§9.1).
use std::hash::Hasher;

// Keys are guest addresses under the embedder's control. HashDoS resistance is
// out of scope; avoid SipHash's cost on every shared dispatch lookup.
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
