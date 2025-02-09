from random import randint, choice
from typing import List, Optional, Tuple, Dict
__MAX_VAL__ = 0xffffffff

class reg_file:
    aliases = {'fp': 'r11', 'sp': 'r13', 'lr': 'r14', 'pc': 'r15'}
    def __init__(self):
        self.regs = {f'r{_}': randint(0, __MAX_VAL__) for _ in range(16)}

    def set(self, reg: str, val: int) -> None:
        if reg == None:
            return
        reg = reg.lower()
        if reg in self.aliases:
            reg = self.aliases[reg]
        if reg not in self.regs:
            raise RuntimeError(f'Invalid register: {reg}')
        converted_val = val & __MAX_VAL__
        assert converted_val in range(0, __MAX_VAL__ + 1)
        self.regs[reg] = converted_val

    def get(self, reg: str) -> int:
        if reg == None:
            raise RuntimeError(f'Error: trying to read null register')
        reg = reg.lower()
        if reg in self.aliases:
            reg = self.aliases[reg]
        if reg not in self.regs:
            raise RuntimeError(f'Invalid register: {reg}')
        return self.regs[reg]

    def gen_reg_list(self, regs: Optional[List[str]]=None) -> List[Tuple[str, int]]:
        s = []
        if regs == None:
            regs = self.regs
        for reg in regs:
            try:
                s.append((reg, self.get(reg)))
            except:
                s.append((reg, f'{reg} not found'))
        return s

class alu:
    ALU_OPS = ['and', 'eor', 'sub', 'rsb', 'add', 'adc', 'sbc', 'rsc', 'tst', 'teq', 'cmp', 'cmn', 'orr', 'mov', 'bic', 'mvn', 'lsl', 'lsr', 'asr', 'ror']
    status = {_:False for _ in ['gt', 'ge', 'lt', 'le', 'eq', 'ne']}
    status[''] = True
    def eval_op(instr_name, val1, val2) -> Optional[int]:
        val1 = val1 & __MAX_VAL__ # to handle negatives
        # note that mov and mvn use val2 rather than val1 as their primary argument
        assert val1 >= 0 and val1 <= __MAX_VAL__
        if val2:
            val2 = val2 & __MAX_VAL__
            assert val2 >= 0 and val2 <= __MAX_VAL__
        result = None
        if instr_name == 'and':
            result = val1 & val2
        elif instr_name == 'eor':
            result = val1 ^ val2
        elif instr_name == 'sub':
            result = val1 - val2
        elif instr_name == 'rsb':
            result = val2 - val1
        elif instr_name == 'add':
            result = val1 + val2
      # elif instr_name == 'adc':
      #     result = (maybe implement later)
      # elif instr_name == 'sbc':
      #     result = (maybe implement later)
      # elif instr_name == 'rsc':
      #     result = (maybe implement later)
      # elif instr_name == 'tst':
      #     result = (maybe implement later)
      # elif instr_name == 'teq':
      #     result = (maybe implement later)
        elif instr_name == 'cmp':
            # for signed comparisons, make sure to adjust val1 and val2 to be negative based on 2s comp value
            val1_signed_value = val1
            val2_signed_value = val2
            if val1 >= 0x80000000:
                val1_signed_value -= 0x100000000
            if val2 >= 0x80000000:
                val2_signed_value -= 0x100000000
            alu.status['gt'] = (val1_signed_value >  val2_signed_value)
            alu.status['ge'] = (val1_signed_value >= val2_signed_value)
            alu.status['lt'] = (val1_signed_value <  val2_signed_value)
            alu.status['le'] = (val1_signed_value <= val2_signed_value)
            alu.status['eq'] = (val1 == val2)
            alu.status['ne'] = (val1 != val2)
            return None
      # elif instr_name == 'cmn':
      #     result = (maybe implement later)
        elif instr_name == 'orr':
            result = val1 | val2
        elif instr_name == 'mov':
            result = val2
        elif instr_name == 'bic':
            result = val1 & (~val2)
        elif instr_name == 'mvn':
            result = ~val2
        # maybe move the shifts out later to properly simulate all instrs having optional shifts
        elif instr_name == 'lsl':
            if val2 > 32:
                result = 0
            else:
                result = val1 << val2
        elif instr_name == 'lsr':
            if val2 > 32:
                result = 0
            else:
                result = val1 >> val2
        elif instr_name == 'asr':
            sign_bit = (val1 & (0x80000000)) >> 31
            if val2 > 32:
                result = __MAX_VAL__ * sign_bit
            else:
                # if sign bit is 1, sign_extension has the leading 1s, else it is 0
                # get the leading 1s by generating a mask of bottom 32-val2 bits, then subtracting from the total 32bit mask
                sign_extension = sign_bit * (__MAX_VAL__ - ((1 << (32 - val2)) - 1))
                result = sign_extension | (val1 >> val2)
        elif instr_name == 'ror':
            val2 = val2 & 0x1f # mod 32
            result = (val1 >> val2) | (val1 << (32 - val2))
        else:
            raise NotImplementedError(f"Instruction `{instr_name}` not implemented.")
        # truncate result back to 32 bits
        result = result & __MAX_VAL__
        return result

class data:
    # default hardcoded to little endian
    def __init__(self):
        self.mem = dict()

    def format(self, format_type='Hex'):
        if not len(self.mem):
            return ''
        # at least 1 addr exists, get sorted list of words
        addrs = sorted(list({(_ & ~3) for _ in self.mem.keys()}), reverse=True)
        max_len = len(hex(addrs[0])) - 2 # get length of biggest address
        lines = []
        if format_type == 'Dec':
            header = ' ' * (5 + max_len) + ' .3  .2  .1  .0'
        elif format_type == 'Char':
            header = ' ' * (5 + max_len) + '  .3   .2   .1   .0'
        else:
            # eventually handle exception here
            header = ' ' * (5 + max_len) + '.3 .2 .1 .0'
        lines.append(header)
        prev_addr = None
        for addr in addrs:
            if prev_addr != None and prev_addr != addr +4:
                lines.append('...')
            prev_addr = addr
            line = f'0x{hex(addr)[2:].zfill(max_len)}: |'
            for i in range(3, -1, -1):
                byte = addr + i
                if byte in self.mem:
                    if format_type == 'Dec':
                        line += f'{self.mem[byte]:3d}|'
                    elif format_type == 'Char':
                        byte = self.mem[byte]
                        if byte == 0:
                            formatted = f"'\\0'"
                        elif byte not in range(32, 127):
                            formatted = f'0x{byte:02x}'
                        else:
                            formatted = f" '{byte:c}'"
                        line += f'{formatted}|'
                    else:
                        line += f'{self.mem[byte]:02x}|'
                else:
                    if format_type == 'Dec':
                        line += '-'
                    elif format_type == 'Char':
                        line += '--'
                    line += '--|'
            lines.append(line)
        return '\n'.join(lines)

    def ldr(self, addr: int) -> int:
        if addr & 0x3 != 0:
            raise ValueError(f"Address 0x{addr:08x} is misaligned for ldr")
        word = 0
        for i in range(4):
            load_addr = addr + i
            shift_amt = 8 * i
            if load_addr not in self.mem:
                # just pretend it has random byte and add to memory
                self.mem[load_addr] = randint(0, 0xff)
            word |= (self.mem[load_addr] << shift_amt)
        return word

    def ldrh(self, addr: int) -> int:
        if addr & 0x1 != 0:
            raise ValueError(f"Address 0x{addr:08x} is misaligned for ldrh")
        hword = 0
        for i in range(2):
            load_addr = addr + i
            shift_amt = 8 * i
            if load_addr not in self.mem:
                # just pretend it has random byte and add to memory
                self.mem[load_addr] = randint(0, 0xff)
            hword |= (self.mem[load_addr] << shift_amt)
        return hword

    def ldrb(self, addr: int) -> int:
        if addr not in self.mem:
            # just pretend it has random byte and add to memory
            self.mem[addr] = randint(0, 0xff)
        byte = self.mem[addr]
        return byte

    def ldrsh(self, addr: int) -> int:
        if addr & 0x1 != 0:
            raise ValueError(f"Address 0x{addr:08x} is misaligned for ldrsh")
        hword = 0
        for i in range(2):
            load_addr = addr + i
            shift_amt = 8 * i
            if load_addr not in self.mem:
                # just pretend it has random byte and add to memory
                self.mem[load_addr] = randint(0, 0xff)
            hword |= (self.mem[load_addr] << shift_amt)
        word = hword
        if hword & 0x8000:
            word |= 0xffff0000
        return word

    def ldrsb(self, addr: int) -> int:
        if addr not in self.mem:
            # just pretend it has random byte and add to memory
            self.mem[addr] = randint(0, 0xff)
        byte = self.mem[addr]
        word = byte
        if byte & 0x80:
            word |= 0xffffff00
        return word

    def str(self, addr: int, word: int) -> None:
        if addr & 0x3 != 0:
            raise ValueError(f"Address 0x{addr:08x} is misaligned for str")
        for i in range(4):
            store_addr = addr + i
            shift_amt = 8 * i
            self.mem[store_addr] = (word >> shift_amt) & 0xff

    def strh(self, addr: int, hword: int) -> None:
        if addr & 0x1 != 0:
            raise ValueError(f"Address 0x{addr:08x} is misaligned for strh")
        for i in range(2):
            store_addr = addr + i
            shift_amt = 8 * i
            self.mem[store_addr] = (hword >> shift_amt) & 0xff

    def strb(self, addr: int, byte: int) -> None:
        self.mem[addr] = byte & 0xff

class top_level:
    def __init__(self, init_regs=None):
        rf = reg_file()
        self.branch_table = dict()
        if init_regs:
            for reg in init_regs:
                rf.set(reg, init_regs[reg])
        rf.set('pc', 0) # init pc
        # init sp
        sp_value = choice(range(0xfffd0000, 0xffff0000, 8))
        rf.set('sp', sp_value)
        # init fp
        rf.set('fp', sp_value + choice(range(0x40, 0x60, 0x8)))
        self.rf = rf
        self.alu = alu()
        self.files = {'': ''} # default empty file when call stack over
        self.data = data()
        self.debug_mode = False
        self.current_file = None
        self.global_branch_table = dict()
        self.call_stack = ['']
        self.function_call = [] # used for tracing function calls

    def add_file(self, filename, instrs, is_main=False):
        instrs, self.line_num_map = preprocess(instrs) # line_num_map doesn't currently work due to labels getting trailing newlines
        self.branch_table[filename] = dict()
        self.files[filename] = instrs
        if is_main:
            self.current_file = filename
            self.instrs = instrs
        self.resolve_branch_table(filename, instrs)

    def resolve_branch_table(self, filename, instrs) -> None:
        for i in range(len(instrs)):
            instr = instrs[i]
            stripped = instr.strip()
            if stripped.endswith(':'):
                name = stripped[:-1]
                # we have a branch
                if name in self.branch_table[filename]:
                    raise RuntimeError(f'label {stripped} appears multiple times in {filename}')
                self.branch_table[filename][name] = i
                if not name.startswith('.L'):
                    # we have a global branch
                    if name in self.global_branch_table:
                        raise RuntimeError(f'label {stripped} appears multiple times, if you are trying to put it in multiple files then prefix it with `.L` to mark it as local')
                    self.global_branch_table[name] = (filename, i)

    def step_instr(self):
        # returns bool or debug log string
        pc = self.rf.get('pc') >> 2 # we want to pretend pc is incrementing 4 bytes at a time
        while pc < len(self.instrs):
            # just skip all labels
            instr = self.instrs[pc]
            if not instr.endswith(':'):
                break
            pc += 1
        self.rf.set('pc', pc << 2)
        if pc >= len(self.instrs):
            return False
        if instr.startswith('.'):
            instr = instr[1:]
            # emulator directive
            if instr.startswith('break'):
                toks = instr.split()[1:]
                if len(toks) == 0:
                    return_val = ''
                # if there is a condition, break if condition is True
                elif eval_cond(self.rf, toks):
                    return_val = ''
                else:
                    return_val = True
            elif instr.startswith('toggle-debug'):
                toks = instr.split()[1:]
                if len(toks) == 0:
                    self.debug_mode = not self.debug_mode
                    return_val = f'Debugging mode set to {self.debug_mode}\n'
                # if there is a condition, flip debug_mode if condition is True
                elif eval_cond(self.rf, toks):
                    self.debug_mode = not self.debug_mode
                    return_val = f'Debugging mode set to {self.debug_mode}\n'
                else:
                    return_val = True
            elif instr.startswith('dump-reg'):
                args = instr.split()
                regs = args[1:]
                if len(regs) == 0:
                    return_val = str(self.rf.gen_reg_list()) + '\n'
                else:
                    actual_regs = [reg.replace('.h', '') for reg in regs]
                    in_hex = {i for i, r in enumerate(regs) if r.endswith('.h')} # the regs that got .h stripped
                    reg_list = self.rf.gen_reg_list(actual_regs)
                    for idx, tup in enumerate(reg_list):
                        if idx in in_hex:
                            reg_list[idx] = (tup[0], hex(tup[1]))
                    return_val = str(reg_list) + '\n'
            elif instr == 'dump-alu':
                return_val = str(self.alu.status) + '\n'
            elif instr.startswith('print'):
                instr = instr[5:].strip()
                if instr:
                    return_val = f'{instr}\n'
                else:
                    return_val = 'Warning: Empty print statement\n'
            else:
                raise RuntimeError(f'Unknown directive `{instr}`')
            pc += 1
            self.rf.set('pc', pc << 2)
            return return_val
        # tokenize the instruction
        toks = instr.strip().replace(',', ' ').split()
        for i in range(len(toks)):
            # lowercase if not a label
            if toks[i] not in self.branch_table[self.current_file]:
                toks[i] = toks[i].lower()
        if toks[0] in alu.ALU_OPS:
            # alu type
            rd = toks[1]
            rm = toks[-1]
            # evaluate rm to get a value
            if rm.startswith('r'):
                rm = self.rf.get(rm)
            else:
                rm = int(rm, 0)
            if toks[0] == 'cmp': # TODO generalize to multiple CPSR ops
                if len(toks) != 3:
                    raise RuntimeError(f'Unable to parse `{instr}`, expected 2 arguments to {toks[0]}')
                rn = self.rf.get(rd)
                rd = None
            elif len(toks) == 4:
                # op rd rn rm
                # get rn value
                rn = self.rf.get(toks[2])
            elif len(toks) == 3:
                # op rd rm
                # either a mov rd rm or using the add r4 r1 syntax, treat rn as rd
                rn = self.rf.get(rd)
            else:
                raise RuntimeError(f'Unable to parse `{instr}`, found {len(toks)} tokens, expected 3 or 4')
            self.rf.set(rd, alu.eval_op(toks[0], rn, rm))
        elif toks[0].startswith('b'):
            branch_table = self.branch_table[self.current_file]
            # branch
            cond = toks[0][1:]
            if cond == 'al':
                cond = ''
            if len(toks) != 2:
                raise RuntimeError(f'Invalid number of arguments in `{instr}`')
            if cond == 'l':
                self.call_stack.append(self.current_file) # so we can go back on bx call
                self.function_call.append(self.current_file) # adding to function call stack
                if toks[1] not in self.global_branch_table:
                    raise RuntimeError(f'Unknown global label {toks[1]}')
                new_file, i = self.global_branch_table[toks[1]]
                self.current_file = new_file
                self.instrs = self.files[new_file]
                lr = pc + 1
                self.rf.set('lr', (lr << 2)) # emulate word addresses
                pc = i
            elif cond == 'x':
                pc = self.rf.get(toks[1]) >> 2 # byte to word translation, force alignment
                pc -= 1 # because it automatically gets incremented again at end of step_instr, and saved value was already pc+1
                self.current_file = self.call_stack.pop(-1)
                self.instrs = self.files[self.current_file]
            else:
                if cond not in self.alu.status:
                    raise RuntimeError(f'Unknown condition {cond}')
                if toks[1] not in branch_table:
                    raise RuntimeError(f'Unknown local label {toks[1]}')
                if self.alu.status[cond]:
                    pc = branch_table[toks[1]]
        elif toks[0].startswith('ldr') or toks[0].startswith('str'):
            # should be a memory op (TODO support push, pop, figure out what to do about functions)
            if len(toks) == 4:
                # base + offset, combine into addr
                if not (toks[2].startswith('[') and toks[3].endswith(']')):
                    raise RuntimeError(f'Expected square brackets `[` `]` in `{instr}` for memory address')
                base_addr = self.rf.get(toks[2][1:])
                offset = toks[3][:-1]
                if offset.startswith('r'):
                    offset = self.rf.get(offset)
                else:
                    offset = int(offset, 0)
                addr = base_addr + offset
            elif len(toks) == 3:
                # just `addr, read and evaluate
                if not (toks[2].startswith('[') and toks[2].endswith(']')):
                    raise RuntimeError(f'Expected square brackets `[` `]` in `{instr}` for memory address')
                addr = self.rf.get(toks[2][1:-1])
            else:
                raise RuntimeError(f'Unable to resolve instruction `{instr}`')
            if toks[0].startswith('l'):
                # load type
                if toks[0].endswith('b'):
                    result = self.data.ldrb(addr) if 's' not in toks[0] else self.data.ldrsb(addr)
                elif toks[0].endswith('h'):
                    result = self.data.ldrh(addr) if 's' not in toks[0] else self.data.ldrsh(addr)
                elif toks[0] == 'ldr':
                    result = self.data.ldr(addr)
                else:
                    raise RuntimeError(f'Unable to resolve instruction {toks[0]}')
                #print(f'loaded {result} from {addr}')
                self.rf.set(toks[1], result)
            else:
                value = self.rf.get(toks[1])
                if toks[0].endswith('b'):
                    result = self.data.strb(addr, value)
                elif toks[0].endswith('h'):
                    result = self.data.strh(addr, value)
                elif toks[0] == 'str':
                    result = self.data.str(addr, value)
                else:
                    raise RuntimeError(f'Unable to resolve instruction {toks[0]}')
                #print(f'stored {value} to {addr}')
        elif toks[0] == 'push' or toks[0] == 'pop':
            push_or_pop = toks[0]
            if len(toks) == 1 or not toks[1].startswith('{') or not toks[-1].endswith('}'):
                raise RuntimeError(f'Expected nonempty list of registers in brackets `{{` `}}` for {toks[0]}')
            # tokenize this one differently
            toks = instr.strip().replace(',', ' , ').replace('-', ' - ').replace('{', ' ').replace('}', ' ').split()[1:]
            reg_range = False
            reg_start = None
            reg_list_cleaned = []
            for tok in toks:
                if reg_range and self.rf.get(tok) != None:
                    # we have a register we can add to list to end range
                    # get un-aliased name if necessary
                    if tok in reg_file.aliases:
                        tok = reg_file.aliases[tok]
                    end_idx = int(tok[1:])
                    reg_list_cleaned.extend([f'r{_}' for _ in range(reg_start, end_idx + 1)])
                    # done with range
                    reg_range = False
                    reg_start = None
                elif tok == ',':
                    # write the most recent reg if one exists
                    if reg_start != None:
                        reg_list_cleaned.append(f'r{reg_start}')
                    reg_range = False
                    reg_start = None
                elif tok == '-':
                    reg_range = True
                elif self.rf.get(tok) != None:
                    # we have a register we can add to list to end range
                    # make sure reg_start is none
                    if reg_start != None:
                        raise RuntimeError(f'Missing comma before register {tok}')
                    # get un-aliased name if necessary
                    if tok in reg_file.aliases:
                        tok = reg_file.aliases[tok]
                    reg_start = int(tok[1:])
            # flush last one if necessary
            if reg_start != None:
                reg_list_cleaned.append(f'r{reg_start}')
            if len(set(reg_list_cleaned)) != len(reg_list_cleaned):
                raise RuntimeError(f'Duplicate registers in {push_or_pop}: {reg_list_cleaned}')
            # ugly fix to make sure the actual nums are in order
            reg_nums = [int(_[1:]) for _ in reg_list_cleaned]
            if sorted(reg_nums) != reg_nums:
                raise RuntimeError(f'Error: list provided to {push_or_pop} should have registers listed in strictly increasing order, violated in {reg_list_cleaned}')
            # at this point, we can assume the list is in ascending order and has no duplicates
            if push_or_pop == 'push':
                # get reg values
                values = [self.rf.get(reg) for reg in reg_list_cleaned]
                stack_ptr = self.rf.get('sp')
                for i, value in enumerate(reversed(values)):
                    self.data.str(stack_ptr - 4 - (i << 2), value)
                stack_ptr -= (len(values) << 2)
                # behavior of pushing sp does not match ARM spec but is probably the only somewhat sane approach for an otherwise illogical action
                self.rf.set('sp', stack_ptr)
            else:
                # get mem values
                stack_ptr = self.rf.get('sp')
                values = [self.data.ldr(stack_ptr + (i << 2)) for i in range(len(reg_list_cleaned))]
                for i, value in enumerate(values):
                    self.rf.set(reg_list_cleaned[i], value)
                # behavior of popping sp does not match ARM spec but is probably the only somewhat sane approach for an otherwise illogical action
                stack_ptr += (len(values) << 2)
                self.rf.set('sp', stack_ptr)
        else:
            raise RuntimeError(f'Unknown instruction `{instr}`')
        pc += 1
        self.rf.set('pc', pc << 2)
        return True if not self.debug_mode else f'Executed `{instr}` -> next pc: {pc}\n'

def eval_cond(rf, toks):
    result = True
    for tok in toks:
        args = tok.split('==')
        if len(args) != 2:
            raise RuntimeError(f'Trying to parse token {tok}, expected format a==b')
        v1 = rf.get(args[0])
        try:
            v2 = rf.get(args[1])
        except:
            v2 = int(args[1], 0)
        result = result and (v1 == v2)
    return result

def preprocess(instrs: str) -> List[str]:
    no_comments = []
    state = None
    for char in instrs:
        if state == None:
            if char == '/':
                state = 'singleslash'
            elif char == ';' or char == '@':
                state = 'linecomment'
            else:
                no_comments.append(char)
        elif state == 'singleslash':
            if char == '/':
                state = 'linecomment'
            elif char == '*':
                state = 'blockcomment'
            else:
                state = None
                no_comments.append('/')
                no_comments.append(char)
        elif state == 'linecomment':
            if char == '\n':
                no_comments.append(char)
                state = None
        elif state == 'blockcomment':
            if char == '\n':
                no_comments.append(char)
            elif char == '*':
                state = 'blockstar'
        elif state == 'blockstar':
            if char == '\n':
                no_comments.append('\n')
                state = 'blockcomment'
            elif char == '/':
                state = None
            elif char == '*':
                state = 'blockstar'
            else:
                state = 'blockcomment'
    if state == 'blockcomment' or state == 'blockstar':
        raise RuntimeError('Unterminated block comment at end of file')
    instrs = ''.join(no_comments)
    a = []
    instrs = instrs.replace('#', '').replace(':', ':\n').strip()
    instrs = instrs.split('\n')
    line_num_map = {} # used for keeping track of error line numbers, map the pc number in stripped code back to actual file line number (currently broken)
    for i, instr in enumerate(instrs):
        if 'ldr ' in instr and '=' in instr:
            instr = instr.replace('ldr', 'mov').replace('=', '')
        instr = instr.strip()
        if instr:
            a.append(instr)
            line_num_map[len(a)] = i + 1 # line numbers in editor are 1-indexed
    return a, line_num_map
