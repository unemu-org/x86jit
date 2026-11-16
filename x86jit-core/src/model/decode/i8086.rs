//! 8086 Family UM §4.2 Tables 4-12/4-13, pp. 4-22–36 (page images checked).
use super::{DecodeAction, DecodeSet};
use iced_x86::{Instruction, Mnemonic, Register};

pub(super) const SET: DecodeSet = DecodeSet {
    models: 4,
    options: iced_x86::DecoderOptions::NO_INVALID_CHECK,
    absent: None,
    legacy_segments_only: true,
};

// NEC V20/V30 native mode: the 8086 rows plus the 80186-class rows of NEC UM
// §12 (bit 3). Like the 8086 it has no invalid-opcode interrupt (§6 p. 6-1
// assigns vectors 0-5 only), so every unlisted form is a typed refusal.
pub(super) const NEC_SET: DecodeSet = DecodeSet {
    models: 4 | 8,
    options: iced_x86::DecoderOptions::NO_INVALID_CHECK,
    absent: None,
    legacy_segments_only: true,
};

pub(crate) fn prefix(byte: u8) -> bool {
    matches!(byte, 0x26 | 0x2e | 0x36 | 0x3e | 0xf0 | 0xf2 | 0xf3)
}

// Table 4-13 is the byte-level inventory. Unlike later processors there is
// no invalid-opcode exception; every unlisted form is a typed refusal.
pub(super) fn decode(bytes: &[u8], insn: &Instruction) -> DecodeAction {
    decode_model::<false>(bytes, insn)
}

// NEC UM §12 and Appendix A pp. A-1–3 are the V20/V30 inventory.
pub(super) fn decode_nec(bytes: &[u8], insn: &Instruction) -> DecodeAction {
    decode_model::<true>(bytes, insn)
}

fn decode_model<const NEC: bool>(bytes: &[u8], insn: &Instruction) -> DecodeAction {
    let absent = DecodeAction::Absent(None);
    let first = bytes.iter().position(|b| !prefix(*b));
    let Some(first) = first else { return absent };
    let b = &bytes[first..];
    let Some(&opcode) = b.first() else {
        return absent;
    };
    // UM §2.5 p. 2-17: "In the case of a locked repeated string instruction,
    // LOCK does remain active for the duration of the block operation"; NEC
    // UM p. 12-173 says the same of BUSLOCK. The engine has no primitive that
    // holds the bus for a whole block operation: a typed refusal. Other locked
    // forms are decided after lowering (`lift::i8086::locked`).
    if bytes[..first].contains(&0xf0) && bytes[..first].iter().any(|b| matches!(b, 0xf2 | 0xf3)) {
        return absent;
    }
    if if NEC {
        known_absent_nec(bytes)
    } else {
        known_absent(bytes)
    } {
        return absent;
    }
    // The fetch decodes the NEC 0Fh and FPO2 forms itself (decision-23 C4b);
    // the iced tables describe other instructions at these bytes.
    if NEC && matches!(opcode, 0x0f | 0x66 | 0x67) {
        return absent;
    }
    // UM §4.2 Table 4-12 p. 4-27 defines the whole ESC encoding space.
    if matches!(opcode, 0xd8..=0xdf) {
        return if bytes[..first].iter().any(|b| matches!(b, 0xf2 | 0xf3)) {
            absent
        } else {
            DecodeAction::Defined
        };
    }
    if insn.is_invalid()
        || super::MODELS_BY_CODE
            .get(insn.code() as usize)
            .copied()
            .unwrap_or(0)
            & if NEC { NEC_SET.models } else { SET.models }
            == 0
    {
        return absent;
    }
    let extension = b.get(1).map(|b| (b >> 3) & 7);
    let valid = match opcode {
        // Decision-23's accepted compatibility amendment: SDM Vol 3B §25.15
        // pp. 25-5–6 defines 82h as 80h and F6h/F7h /1 as TEST /0;
        // oracles/ss8088/v2/metadata.json marks these forms "alias".
        // The amendment also admits ordinary 83h /1,/4,/6: SDM Vol 2A
        // §3.3 AND p. 3-60, Vol 2B §4.3 OR p. 4-163, and
        // Vol 2D §6.1 XOR p. 6-40; the same metadata marks them
        // "normal", although UM Table 4-13 p. 4-31 omits them.
        0x83 => extension.is_some(),
        0x8c => matches!(extension, Some(0..=3)),
        // 286 PRM App. C item 6 p. C-2 explicitly calls MOV CS undefined.
        0x8e => matches!(extension, Some(0 | 2 | 3)),
        0x8f | 0xc6 | 0xc7 => extension == Some(0),
        0xc4 | 0xc5 => b.get(1).is_some_and(|b| b & 0xc0 != 0xc0),
        // SDM Vol 3B §25.15 pp. 25-5–6 also lists /6, but
        // oracles/ss8088/v2/metadata.json marks D0–D3 /6 "undocumented";
        // the amendment therefore keeps these forms as typed refusals.
        0xd0..=0xd3 => extension.is_some_and(|e| e != 6),
        // Table 4-12 p. 4-24 fixes the second byte to 00001010.
        0xd4 | 0xd5 => b.get(1) == Some(&10),
        0xfe => matches!(extension, Some(0 | 1)),
        0xff => extension.is_some_and(|e| e != 7),
        _ => true,
    } && (!NEC || nec_admits(b));
    if !valid
        || (0..insn.op_count()).any(|i| matches!(insn.op_register(i), Register::FS | Register::GS))
    {
        return absent;
    }
    // UM §2.7 pp. 2-42–43 defines REP for MOVS/STOS/LODS and REPE/REPNE
    // for CMPS/SCAS. Other prefix/instruction combinations remain refused.
    // NEC UM p. 12-15 adds REP (F3h) for OUTM and INM; REPNE stays a
    // block-comparison prefix only.
    let port_string = matches!(
        insn.mnemonic(),
        Mnemonic::Insb | Mnemonic::Insw | Mnemonic::Outsb | Mnemonic::Outsw
    );
    if bytes[..first].iter().any(|b| matches!(b, 0xf2 | 0xf3))
        && !matches!(
            insn.mnemonic(),
            Mnemonic::Movsb
                | Mnemonic::Movsw
                | Mnemonic::Stosb
                | Mnemonic::Stosw
                | Mnemonic::Lodsb
                | Mnemonic::Lodsw
                | Mnemonic::Cmpsb
                | Mnemonic::Cmpsw
                | Mnemonic::Scasb
                | Mnemonic::Scasw
        )
        && !(NEC && port_string)
    {
        return absent;
    }
    // NEC UM p. 12-28: INM's destination is always DS1 and "a segment
    // override prefix is prohibited"; its effect is not described.
    if NEC
        && matches!(insn.mnemonic(), Mnemonic::Insb | Mnemonic::Insw)
        && bytes[..first]
            .iter()
            .any(|b| matches!(b, 0x26 | 0x2e | 0x36 | 0x3e))
    {
        return absent;
    }
    if bytes[..first].contains(&0xf2)
        && !matches!(
            insn.mnemonic(),
            Mnemonic::Cmpsb | Mnemonic::Cmpsw | Mnemonic::Scasb | Mnemonic::Scasw
        )
    {
        return absent;
    }
    if let Some(alias) = super::defined_alias(insn.code()) {
        DecodeAction::Alias(alias)
    } else {
        DecodeAction::Defined
    }
}

// NEC V20/V30 forms beyond the 8086 table, and the 8086 forms whose result
// the NEC manual does not state. Called after the shared 8086 checks.
fn nec_admits(b: &[u8]) -> bool {
    let modrm = b.get(1).copied();
    match b[0] {
        // NEC UM p. 12-147: "(SP - 1, SP - 2) <- reg16, SP <- SP - 2" and
        // p. 12-150 "reg16 <- (SP + 1, SP), SP <- SP + 2" do not say which
        // SP value PUSH SP stores or what POP SP leaves: typed refusals.
        0x54 | 0x5c => false,
        0x8f => modrm != Some(0xc4),
        0xff => modrm != Some(0xf4),
        // NEC UM p. 12-170: CHKIND reg16,mem32 has only a memory operand.
        0x62 => modrm.is_some_and(|m| m & 0xc0 != 0xc0),
        // Shift group /6 stays refused (decision-23 compatibility amendment).
        0xc0 | 0xc1 => modrm.is_some_and(|m| (m >> 3) & 7 != 6),
        _ => true,
    }
}

// NEC V20/V30 byte table: the 8086 rows, the 80186-class forms of NEC UM §12
// (60h-62h, 68h-6Fh, C0h/C1h, C8h/C9h), FPO2 (66h/67h, pp. 12-174–175) and
// the 0Fh forms of `known_absent_nec_0f`. 63h, D6h and F1h are typed
// refusals. 64h/65h (REPNC/REPC, p. 12-14) are prefixes, which the fetch
// takes before this table; as an opcode they are absent.
pub(super) fn known_absent_nec(bytes: &[u8]) -> bool {
    let b = &bytes[bytes.iter().take_while(|b| prefix(**b)).count()..];
    let Some(&opcode) = b.first() else {
        return false;
    };
    if !matches!(opcode, 0x00..=0x53 | 0x55..=0x5b | 0x5d..=0x62 | 0x66..=0xd5 | 0xd7..=0xef | 0xf4..=0xff)
    {
        return true;
    }
    let Some(&second) = b.get(1) else {
        return false;
    };
    match opcode {
        0x0f => known_absent_nec_0f(&b[1..]),
        0x66 | 0x67 => false,
        0x62 => second & 0xc0 == 0xc0,
        0xc0 | 0xc1 => (second >> 3) & 7 == 6,
        0x8f => second == 0xc4 || known_absent(b),
        0xff => second == 0xf4 || known_absent(b),
        0x60 | 0x61 | 0x68..=0x6f | 0xc8 | 0xc9 => false,
        _ => known_absent(b),
    }
}

// The NEC 0Fh map after its first byte (NEC UM §12):
// - 10h-1Fh TEST1/CLR1/SET1/NOT1 (pp. 12-81–98) and 28h/2Ah ROL4/ROR4
//   (pp. 12-45–46): ModRM bits 5:3 are 000; ROL4/ROR4 AL is refused, since AL
//   would be both the rotated byte and the digit source and the page gives
//   no result for that;
// - 20h/22h/26h ADD4S/SUB4S/CMP4S (pp. 12-42–44): no ModRM;
// - 31h/33h INS/EXT reg,reg (pp. 12-21, 12-23): ModRM 11 reg2 reg1;
//   39h/3Bh INS/EXT reg,imm4 (pp. 12-22, 12-24): 11 000 reg, imm4 0-15. EXT
//   with AL or AH as the offset register is refused. P. 12-23 orders the
//   update ("AW <- 16-bit field ... After the transfer, the IX register and
//   the 8-bit register specified by the first operand are updated"), so the
//   page would leave the new offset there, but the hardware corpus
//   documentation (oracles/ssv20/README.md: "Using AL or AH as the first
//   operand to EXT is invalid") marks the form invalid, and that derived
//   result is not claimed. INS AL/AH differs: AW is only read, before the
//   transfer, and AL/AH is written once, after it;
// - FFh BRKEM (p. 12-169).
// Every other second byte is a typed refusal.
fn known_absent_nec_0f(b: &[u8]) -> bool {
    let second = b[0];
    let modrm = b.get(1).copied();
    match second {
        0x10..=0x1f | 0x28 | 0x2a => {
            modrm.is_some_and(|m| m & 0x38 != 0 || (matches!(second, 0x28 | 0x2a) && m == 0xc0))
        }
        0x31 | 0x33 => modrm.is_some_and(|m| m & 0xc0 != 0xc0 || (second == 0x33 && m & 3 == 0)),
        0x39 | 0x3b => {
            modrm.is_some_and(|m| m & 0xf8 != 0xc0 || (second == 0x3b && m & 3 == 0))
                || b.get(2).is_some_and(|imm| *imm > 15)
        }
        0x20 | 0x22 | 0x26 | 0xff => false,
        _ => true,
    }
}

// Refuse as soon as the byte table establishes absence, before fetching an
// unrelated missing displacement/immediate byte (decision-23 §3).
pub(super) fn known_absent(bytes: &[u8]) -> bool {
    let b = &bytes[bytes.iter().take_while(|b| prefix(**b)).count()..];
    let Some(&opcode) = b.first() else {
        return false;
    };
    if !matches!(opcode, 0x00..=0x0e | 0x10..=0x5f | 0x70..=0xbf | 0xc2..=0xc7 | 0xca..=0xcf | 0xd0..=0xd5 | 0xd7..=0xef | 0xf4..=0xff)
    {
        return true;
    }
    let Some(&modrm) = b.get(1) else { return false };
    let e = (modrm >> 3) & 7;
    match opcode {
        0x8c => e > 3,
        0x8e => !matches!(e, 0 | 2 | 3),
        0x8f | 0xc6 | 0xc7 => e != 0,
        0x8d | 0xc4 | 0xc5 => modrm & 0xc0 == 0xc0,
        0xd0..=0xd3 => e == 6,
        0xd4 | 0xd5 => modrm != 10,
        0xfe => e > 1,
        0xff => e == 7 || (matches!(e, 3 | 5) && modrm & 0xc0 == 0xc0),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use iced_x86::{Code, Decoder};

    // A row marked for the 8086 must be admitted by the policy above for at
    // least one encoding; a row the policy always refuses is outside the set.
    #[test]
    fn every_8086_row_is_admitted_by_some_encoding() {
        let mut admitted = std::collections::HashSet::new();
        for opcode in 0..=255u8 {
            for modrm in 0..=255u8 {
                let bytes = [opcode, modrm, 0, 0, 0, 0];
                let insn = Decoder::new(16, &bytes, SET.options).decode();
                let len = insn.len().max(1);
                if !matches!(decode(&bytes[..len], &insn), DecodeAction::Absent(_)) {
                    admitted.insert(insn.code());
                }
            }
        }
        let refused: Vec<Code> = super::super::ENCODINGS
            .iter()
            .filter(|(code, models)| models & SET.models != 0 && !admitted.contains(code))
            .map(|(code, _)| *code)
            .collect();
        assert!(refused.is_empty(), "{refused:?}");
    }

    // The same for the NEC V20/V30 rows (bits 2 and 3).
    #[test]
    fn every_nec_row_is_admitted_by_some_encoding() {
        let mut admitted = std::collections::HashSet::new();
        for opcode in 0..=255u8 {
            for modrm in 0..=255u8 {
                for prefix in [None, Some(0xf3)] {
                    let mut bytes = Vec::from_iter(prefix);
                    bytes.extend([opcode, modrm, 0, 0, 0, 0]);
                    let insn = Decoder::new(16, &bytes, NEC_SET.options).decode();
                    let len = insn.len().max(1);
                    if !matches!(decode_nec(&bytes[..len], &insn), DecodeAction::Absent(_)) {
                        admitted.insert(insn.code());
                    }
                }
            }
        }
        let refused: Vec<Code> = super::super::ENCODINGS
            .iter()
            .filter(|(code, models)| models & NEC_SET.models != 0 && !admitted.contains(code))
            .map(|(code, _)| *code)
            .collect();
        assert!(refused.is_empty(), "{refused:?}");
    }

    // Decision-23 C4b: the NEC 0Fh map admits exactly the second bytes NEC
    // UM §12 lists, 66h/67h open FPO2 and 64h/65h stay prefixes only; the
    // 8086 table still refuses all of them at the first byte, and the NEC
    // iced path never decodes them.
    #[test]
    fn nec_extension_bytes_and_unchanged_8086_refusals() {
        for opcode in [0x0f, 0x63, 0x64, 0x65, 0x66, 0x67] {
            assert!(known_absent(&[opcode]), "{opcode:#x}");
        }
        for second in 0..=255u8 {
            let listed = matches!(
                second,
                0x10..=0x1f | 0x20 | 0x22 | 0x26 | 0x28 | 0x2a | 0x31 | 0x33 | 0x39 | 0x3b | 0xff
            );
            assert_eq!(!known_absent_nec(&[0x0f, second]), listed, "{second:#x}");
        }
        assert!(!known_absent_nec(&[0x66]) && !known_absent_nec(&[0x67]));
        for opcode in [0x63, 0x64, 0x65] {
            assert!(known_absent_nec(&[opcode]), "{opcode:#x}");
        }
        for bytes in [&[0x66, 0xf4][..], &[0x67, 0x90], &[0x0f, 0x10, 0xc0]] {
            let insn = Decoder::new(16, bytes, NEC_SET.options).decode();
            assert!(matches!(decode_nec(bytes, &insn), DecodeAction::Absent(_)));
        }
    }

    // xj-50w.12: UM §2.5 p. 2-17 and NEC UM p. 12-173 hold the bus lock
    // through a whole repeated string, which the engine refuses. LOCK alone
    // reaches lowering (`lift::i8086::locked`), in either prefix order.
    #[test]
    fn lock_is_refused_only_with_a_repeat_prefix() {
        for bytes in [
            &[0xf0, 0xff, 0x07][..],
            &[0xf0, 0x26, 0x01, 0x07],
            &[0x26, 0xf0, 0x01, 0x07],
            &[0xf0, 0x90],
            &[0xf0, 0xa4],
        ] {
            let insn = Decoder::new(16, bytes, SET.options).decode();
            assert!(insn.has_lock_prefix(), "{bytes:02x?}");
            assert_eq!(decode(bytes, &insn), DecodeAction::Defined, "{bytes:02x?}");
            assert_eq!(
                decode_nec(bytes, &insn),
                DecodeAction::Defined,
                "{bytes:02x?}"
            );
        }
        for bytes in [
            &[0xf0, 0xf3, 0xa4][..],
            &[0xf3, 0xf0, 0xa5],
            &[0xf0, 0xf2, 0xa6],
            &[0xf0, 0x26, 0xf3, 0xaa],
        ] {
            let insn = Decoder::new(16, bytes, SET.options).decode();
            assert!(matches!(decode(bytes, &insn), DecodeAction::Absent(None)));
            assert!(matches!(
                decode_nec(bytes, &insn),
                DecodeAction::Absent(None)
            ));
        }
    }

    // Every byte the NEC table refuses at the opcode is refused by the
    // decoder too, and the 8086 policy admits none of the 80186-class rows.
    #[test]
    fn nec_and_8086_byte_tables_agree_with_their_decoders() {
        for opcode in 0..=255u8 {
            let bytes = [opcode, 0xc0, 0, 0, 0, 0];
            if known_absent_nec(&bytes[..1]) {
                let insn = Decoder::new(16, &bytes, NEC_SET.options).decode();
                let len = insn.len().max(1);
                assert!(matches!(
                    decode_nec(&bytes[..len], &insn),
                    DecodeAction::Absent(_)
                ));
            }
        }
        for code in super::super::ENCODINGS
            .iter()
            .filter(|(_, models)| models & 8 != 0)
            .map(|(code, _)| *code)
        {
            assert_eq!(
                super::super::MODELS_BY_CODE[code as usize] & SET.models,
                0,
                "{code:?}"
            );
        }
    }
}
