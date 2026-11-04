//! Guest GPR and arithmetic-flag synchronization (§8.2.3, §12 M5-T3e).
//! Compile-time may-be-dirty analysis over the emitted CFG. A field
//! needs a store if any incoming path wrote it since its last synchronization.

use super::*;

enum Event {
    Write(usize),
    Store(usize, ir::Inst),
    Reload,
}

#[derive(Default)]
pub(super) struct GprSync {
    events: HashMap<Block, Vec<Event>>,
}

impl GprSync {
    pub(super) fn write(&mut self, block: Block, index: usize) {
        self.events
            .entry(block)
            .or_default()
            .push(Event::Write(index));
    }

    pub(super) fn store(&mut self, block: Block, index: usize, inst: ir::Inst) {
        self.events
            .entry(block)
            .or_default()
            .push(Event::Store(index, inst));
    }

    pub(super) fn reload(&mut self, block: Block) {
        self.events.entry(block).or_default().push(Event::Reload);
    }

    pub(super) fn remove_clean_stores(self, func: &mut ir::Function) {
        let blocks: Vec<_> = func.layout.blocks().collect();
        let mut incoming: HashMap<Block, u32> = blocks.iter().map(|&b| (b, 0)).collect();
        // Bits 0..16 track GPRs; bits 16..22 track deferred arithmetic flags.
        // Union at joins and iterate through back-edges to a fixed point. Cold
        // exits have their own blocks, so their stores never clean the hot path.
        loop {
            let mut changed = false;
            for &block in &blocks {
                let mut dirty = incoming[&block];
                for event in self.events.get(&block).into_iter().flatten() {
                    match *event {
                        Event::Write(i) => dirty |= 1 << i,
                        Event::Store(i, _) => dirty &= !(1 << i),
                        Event::Reload => dirty &= !0xffff,
                    }
                }
                let last = func
                    .layout
                    .last_inst(block)
                    .expect("terminated codegen block");
                for dest in func.dfg.insts[last].branch_destination(&func.dfg.jump_tables) {
                    let input = incoming
                        .get_mut(&dest.block(&func.dfg.value_lists))
                        .unwrap();
                    let merged = *input | dirty;
                    changed |= merged != *input;
                    *input = merged;
                }
            }
            if !changed {
                break;
            }
        }
        for (block, events) in self.events {
            let mut dirty = incoming[&block];
            for event in events {
                match event {
                    Event::Write(i) => dirty |= 1 << i,
                    Event::Store(i, inst) => {
                        if dirty & (1 << i) == 0 {
                            func.layout.remove_inst(inst);
                        }
                        dirty &= !(1 << i);
                    }
                    Event::Reload => dirty &= !0xffff,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flag_materialization_requires_every_incoming_path() {
        for both_paths in [false, true] {
            let mut func = ir::Function::new();
            let mut context = FunctionBuilderContext::new();
            let mut builder = FunctionBuilder::new(&mut func, &mut context);
            let entry = builder.create_block();
            let left = builder.create_block();
            let right = builder.create_block();
            let join = builder.create_block();
            let cpu = builder.append_block_param(entry, types::I64);
            let condition = builder.append_block_param(entry, types::I8);
            builder.switch_to_block(entry);
            let value = builder.ins().iconst(types::I8, 1);
            let mut sync = GprSync::default();
            sync.write(entry, 16);
            builder.ins().brif(condition, left, &[], right, &[]);
            for block in [left, right] {
                builder.switch_to_block(block);
                if block == left || both_paths {
                    let inst = builder.ins().store(MemFlags::trusted(), value, cpu, 0);
                    sync.store(block, 16, inst);
                }
                builder.ins().jump(join, &[]);
            }
            builder.switch_to_block(join);
            let at_join = builder.ins().store(MemFlags::trusted(), value, cpu, 0);
            sync.store(join, 16, at_join);
            let repeated = builder.ins().store(MemFlags::trusted(), value, cpu, 0);
            sync.store(join, 16, repeated);
            builder.ins().return_(&[]);
            builder.seal_all_blocks();
            builder.finalize();
            sync.remove_clean_stores(&mut func);
            assert_eq!(func.layout.inst_block(at_join).is_none(), both_paths);
            assert!(func.layout.inst_block(repeated).is_none());
        }
    }
}
