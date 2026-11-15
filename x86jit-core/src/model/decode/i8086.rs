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
    // UM §2.5 pp. 2-17–18 specifies a bus lock for the whole instruction,
    // including an entire REP. This engine has no such bus ownership primitive.
    if bytes[..first].contains(&0xf0) {
        return absent;
    }
    if if NEC {
        known_absent_nec(bytes)
    } else {
        known_absent(bytes)
    } {
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
// (60h-62h, 68h-6Fh, C0h/C1h, C8h/C9h) and BRKEM 0F FF imm8 (p. 12-169).
// 63h-67h, D6h, F1h and the rest of the 0Fh map are typed refusals; 64h/65h
// (REPC/REPNC), 66h/67h (FPO2) and the 0Fh groups are decision-23 C4b.
pub(super) fn known_absent_nec(bytes: &[u8]) -> bool {
    let b = &bytes[bytes.iter().take_while(|b| prefix(**b)).count()..];
    let Some(&opcode) = b.first() else {
        return false;
    };
    if !matches!(opcode, 0x00..=0x53 | 0x55..=0x5b | 0x5d..=0x62 | 0x68..=0xd5 | 0xd7..=0xef | 0xf4..=0xff)
    {
        return true;
    }
    let Some(&second) = b.get(1) else {
        return false;
    };
    match opcode {
        0x0f => second != 0xff,
        0x62 => second & 0xc0 == 0xc0,
        0xc0 | 0xc1 => (second >> 3) & 7 == 6,
        0x8f => second == 0xc4 || known_absent(b),
        0xff => second == 0xf4 || known_absent(b),
        0x60 | 0x61 | 0x68..=0x6f | 0xc8 | 0xc9 => false,
        _ => known_absent(b),
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
