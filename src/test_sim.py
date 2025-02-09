from sim import *

if __name__ != '__main__':
    print('WARNING: this file is for testing purposes only and should NOT be imported anywhere')
    raise ImportError('Importing test-only module test_sim')

is_even_instrs = '''
is_even:
    push {r4-r9, fp, lr}
    add fp, sp, 28

    and r0, r0, 0b1
    eor r0, r0, 0x1

    sub sp, fp, 28
    pop {r4-r9, fp, lr}
    bx lr
'''

func_instrs = '''
func:
    push {r4-r9, fp, lr}
    add fp, sp, 28

    mov r1, 2
    sub sp, sp, 4
    str r1, [sp]

    // return 0
    mov r0, 0

    b .Lend: // skip to end (local label)
    mov r0, 1
.Lend:

    sub sp, fp, 28
    pop {r4-r9, fp, lr}
    bx lr
'''

main_instrs = '''
main:
    push {fp, lr}
    add fp, sp, 4
    bl func

    mov r0, 0
    mov r1, 1
    mov r2, 2
    mov r3, 3

    bl is_even
    .dump-reg r0
    sub sp, fp, 4
    pop {fp, lr}
    bx lr
'''
dut = top_level()
dut.add_file('main.S', main_instrs, is_main=True)
dut.add_file('func.S', func_instrs)
dut.add_file('is_even.S', is_even_instrs)
print(dut.branch_table)
print(dut.global_branch_table)
print(dut.instrs)
print(dut.line_num_map)
r = dut.step_instr()
n_steps = 0
MAX_STEPS = 1e5
while r and n_steps < MAX_STEPS:
    n_steps += 1
    if type(r) == str:
        print(r)
    r = dut.step_instr()
    print(f"{dut.current_file}: {dut.rf.get('pc')}")
formatted_regs = {key: hex(value) for (key,value) in dut.rf.regs.items()}
formatted_mem = dut.data.format()
print(f'register values = {formatted_regs}')
print(f'data values =\n{formatted_mem}')

