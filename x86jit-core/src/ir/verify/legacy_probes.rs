use super::*;
use crate::legacy::{CpuGeneration, LegacyDecodeContext, LegacyOp, LegacyRegime};
use iced_x86::{Code, Encoder, Instruction, OpCodeOperandKind, OpKind, Register};

const IP: u64 = 0x1000;

// Like the compatibility probe, synthesize canonical register and memory forms
// from iced metadata. Keep this witness in core to reach the private legacy lifter.
fn encode_probe(code: Code, bits: u8, memory_operand: Option<u32>) -> Option<Vec<u8>> {
    use OpCodeOperandKind::*;
    let mut instruction = Instruction::default();
    instruction.set_code(code);
    for i in 0..code.op_code().op_count() {
        let kind = code.op_code().op_kind(i);
        if memory_operand == Some(i) || matches!(kind, mem | mem_offs) {
            instruction.set_op_kind(i, OpKind::Memory);
            instruction.set_memory_base(if kind == mem_offs {
                Register::None
            } else if bits == 16 {
                Register::BX
            } else {
                Register::EAX
            });
            instruction.set_memory_displacement64(0x40);
            instruction.set_memory_displ_size(if kind == mem_offs { bits / 8 } else { 1 } as u32);
            continue;
        }
        let register = match kind {
            r8_reg | r8_opcode | r8_or_mem => [Register::AL, Register::CL][i as usize % 2],
            r16_reg | r16_reg_mem | r16_rm | r16_opcode | r16_or_mem => {
                [Register::AX, Register::CX][i as usize % 2]
            }
            r32_reg | r32_reg_mem | r32_rm | r32_opcode | r32_or_mem => {
                [Register::EAX, Register::ECX][i as usize % 2]
            }
            seg_reg | ds => Register::DS,
            es => Register::ES,
            cs => Register::CS,
            ss => Register::SS,
            fs => Register::FS,
            gs => Register::GS,
            cr_reg => Register::CR0,
            dr_reg => Register::DR0,
            tr_reg => Register::TR6,
            al => Register::AL,
            cl => Register::CL,
            ax => Register::AX,
            dx => Register::DX,
            eax => Register::EAX,
            imm8 | imm8_const_1 | imm8sex16 | imm8sex32 => {
                if i > 0 && instruction.op_kind(i - 1) == OpKind::Immediate8 {
                    instruction.set_op_kind(i, OpKind::Immediate8_2nd);
                    instruction.set_immediate8_2nd(1);
                } else {
                    instruction.set_op_kind(i, OpKind::Immediate8);
                    instruction.set_immediate8(1);
                }
                continue;
            }
            imm16 => {
                instruction.set_op_kind(i, OpKind::Immediate16);
                instruction.set_immediate16(1);
                continue;
            }
            imm32 => {
                instruction.set_op_kind(i, OpKind::Immediate32);
                instruction.set_immediate32(1);
                continue;
            }
            br16_1 | br16_2 | br32_1 | br32_4 => {
                instruction.set_op_kind(
                    i,
                    if matches!(kind, br16_1 | br16_2) {
                        OpKind::NearBranch16
                    } else {
                        OpKind::NearBranch32
                    },
                );
                instruction.set_near_branch64(IP + 0x20);
                continue;
            }
            farbr2_2 | farbr4_2 => {
                instruction.set_op_kind(
                    i,
                    if kind == farbr2_2 {
                        OpKind::FarBranch16
                    } else {
                        OpKind::FarBranch32
                    },
                );
                instruction.set_far_branch32((IP + 0x20) as u32);
                instruction.set_far_branch_selector(0);
                continue;
            }
            _ => return Option::None,
        };
        instruction.set_op_kind(i, OpKind::Register);
        instruction.set_op_register(i, register);
    }
    let mut encoder = Encoder::new(u32::from(bits));
    encoder.encode(&instruction, IP).ok()?;
    Some(encoder.take_buffer())
}

#[test]
fn every_legacy_probe_defines_temps_before_use() {
    let mut total = 0;
    for regime in [LegacyRegime::Real, LegacyRegime::Protected] {
        for default_operand_bits in [16, 32] {
            for stack_address_bits in [16, 32] {
                let context = LegacyDecodeContext {
                    generation: CpuGeneration::I386,
                    regime,
                    default_operand_bits,
                    default_address_bits: default_operand_bits,
                    stack_address_bits,
                    cpl: 0,
                    paging_enabled: false,
                };
                let mut checked = 0;
                let mut system_ops = [false; 8];
                for code in Code::values().filter(|code| {
                    let info = code.op_code();
                    *code != Code::INVALID
                        && if default_operand_bits == 16 {
                            info.mode16()
                        } else {
                            info.mode32()
                        }
                }) {
                    let info = code.op_code();
                    let memory_operand = (0..info.op_count()).find(|&i| {
                        matches!(
                            info.op_kind(i),
                            OpCodeOperandKind::r8_or_mem
                                | OpCodeOperandKind::r16_or_mem
                                | OpCodeOperandKind::r32_or_mem
                        )
                    });
                    for form in std::iter::once(None).chain(memory_operand.map(Some)) {
                        let Some(bytes) = encode_probe(code, default_operand_bits, form) else {
                            continue;
                        };
                        let block =
                            match crate::lift::legacy::instruction(&bytes, IP as u32, context) {
                                Ok(block) => block,
                                Err(crate::lift::LiftError::Unsupported { .. }) => continue,
                                Err(error) => panic!("{context:?}/{code:?}/{form:?}: {error:?}"),
                            };
                        assert_eq!(block.legacy_context, Some(context));
                        assert_eq!(
                            block.verify_temp_defs(),
                            Ok(()),
                            "{context:?}/{code:?} memory_operand={form:?}: {:?}",
                            block.ops
                        );
                        for op in &block.ops {
                            let index = match op {
                                IrOp::Legacy(LegacyOp::Table { .. }) => 0,
                                IrOp::Legacy(LegacyOp::ReadControl { .. }) => 1,
                                IrOp::Legacy(LegacyOp::WriteControl { .. }) => 2,
                                IrOp::Legacy(LegacyOp::LoadMsw { .. }) => 3,
                                IrOp::Legacy(LegacyOp::ClearTs) => 4,
                                IrOp::Legacy(LegacyOp::Interrupt { .. }) => 5,
                                IrOp::Legacy(LegacyOp::Iret { .. }) => 6,
                                IrOp::Legacy(LegacyOp::Bound { .. }) => 7,
                                _ => continue,
                            };
                            system_ops[index] = true;
                        }
                        checked += 1;
                    }
                }
                assert!(checked > 1000, "{context:?}: only {checked} blocks checked");
                assert!(
                    system_ops.into_iter().all(|seen| seen),
                    "{context:?}: {system_ops:?}"
                );
                eprintln!("{context:?}: verified {checked} legacy probe blocks");
                total += checked;
            }
        }
    }
    eprintln!("verified {total} legacy probe blocks in total");
}
