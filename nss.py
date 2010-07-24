# coding: utf-8
"""
    nss
    ~~~

    nested style sheets is an implementation of CleverCSS, a
    pythonic approach to CSS developed by Armin Ronacher and Georg Brandl.
"""
import re
from collections import namedtuple, deque
from contextlib import contextmanager
from codecs import open
from fractions import gcd

#: The units supported by nss.
UNITS = [
    '%',  # percentage
    'in', # inch
    'cm', # centimeter
    'mm', # millimeter
    'em', # 1em is equal to the current font size.
    'ex', # 1ex is the x-height of a font (usually half the font-size)
    'pt', # point (1 pt = 1/72 inch)
    'pc', # pica (1 pc = 12 points)
    'px', # pixel
]

missing = object()

_r_number = ur'\d+(?:\.\d+)?'

indent_re = re.compile(ur'^([\t ]*)(.*)$')
comment_re = re.compile(ur'//.*?$')
operator_re = re.compile(ur'({0})'.format(
    u'|'.join(
        map(
            re.escape,
            ['**', '*', '+', '-', '/', '(', ')', '=', ',', ':', '.']
        )
    )
))
number_re = re.compile(ur'({0})'.format(_r_number))
value_re = re.compile(ur'({0}(?:{1}))'.format(_r_number, ur'|'.join(UNITS)))
color_re = re.compile(ur'(#{0})'.format(ur'[A-Fa-f\d]{1,2}' * 3))
string_re = re.compile(ur"""
(
    (?:
        "(?:
            [^"\]]*
            (?:\\.[^"\\]*)*
        )"|
        '(?:
            [^'\\]*
            (?:\\.[^'\\]*)*
        )'
    )|
    [^\s*/();,.+$]+
)
""", re.VERBOSE)
name_re = re.compile(ur"""
(?<!\\)\$?
(?:
    (
        [A-Za-z_\-]
        [A-Za-z0-9_\-]*
    )|
    \{(
        [A-Za-z_\-]
        [A-Za-z0-9_\-]*
    )\}
)
""", re.VERBOSE)

TokenBase = namedtuple('Token', ['id', 'lineno', 'value'])
class Token(TokenBase):
    def __new__(cls, id, lineno, value=missing):
        if value is missing:
            value = id
        return TokenBase.__new__(cls, id, lineno, value)

class LineIterator(object):
    def __init__(self, source, tab_length=8):
        if hasattr(source, 'splitlines'):
            source = source.splitlines()
        self._line_iter = iter(source)
        self.tab_length = tab_length

        self.lineno = 0

    def __iter__(self):
        return self

    def fix_indent(self, line):
        match = indent_re.match(line)
        if match:
            indent = match.group(1)
            fixed_indent = indent.replace(u'\t', u' ' * self.tab_length)
            line = fixed_indent + line[len(indent):]
        return line

    def _readline(self):
        line = u''
        while not line.strip():
            line = comment_re.sub(
                u'',
                self.fix_indent(self._line_iter.next())
            ).rstrip()
            self.lineno += 1
        return line

    def next(self):
        line = u''
        while not line:
            line = self._readline()
            lineno = self.lineno

            comment_start = line.find(u'/*')
            if comment_start > -1:
                stripped_line = line[:comment_start]
                comment_end = line.find(u'*/', comment_start)
                if comment_end >= 0:
                    line = stripped_line + line[comment_end + 2:]
                else:
                    lineno = self.lineno
                    try:
                        while comment_end == -1:
                            line = self._readline()
                            comment_end = line.find(u'*/')
                        line = stripped_line + line[comment_end + 2:]
                    except StopIteration:
                        raise ParserError(
                            self.lineno,
                            'missing end of multiline comment'
                        )
        return lineno, line

        # remove inline comments
        line = comment_re.sub(u'', line)
        return self.lineno, line.rstrip()

class ParserError(Exception):
    def __init__(self, lineno, message):
        Exception.__init__(self, message)
        self.lineno = lineno

    def __str__(self):
        return b'{0} (line {1})'.format(self.args[0], self.lineno)

class SymbolBase(object):
    id = None
    lbp = 0

    def __init__(self, parser, value, lineno):
        self.parser = parser
        self.value = value
        self.lineno = lineno

        self.first = self.second = None
        self._body = None

    @property
    def body(self):
        if self._body is None:
            return filter(None, [self.first, self.second])
        return self._body

    @body.setter
    def body(self, new_body):
        self._body = new_body

    def nud(self):
        raise ParserError(self.lineno, 'Syntax error')

    def led(self, left):
        raise ParserError(
            self.lineno, 'Unknown operator {0}'.format(left.id)
        )

    @classmethod
    def set(cls, name):
        def decorate(func):
            setattr(cls, name, func)
            return func
        return decorate

    def __repr__(self):
        out = [self.id] + (map(repr, self.body) or [self.value])
        return '({0})'.format(' '.join(out))

class StatementBase(SymbolBase):
    scoped = False
    body = None

    def __init__(self, parser, value, lineno):
        SymbolBase.__init__(self, parser, value, lineno)
        self.body = []

    def std(self):
        raise ParserError(self.lineno, 'Syntax error')

    def __repr__(self):
        out = [self.id]
        if self.value:
            out.append(self.value)
        out.extend(map(repr, self.body))
        return '({0})'.format(' '.join(out))

class Root(StatementBase):
    id = 'root'

    def std(self):
        return self

    def __repr__(self):
        result = '({0}'.format(self.id)
        body = ' '.join(map(repr, self.body))
        if body:
            result += ' ' + body
        return result + ')'

class Parser(object):
    def __init__(self):
        self.symbol_table = {}
        self.symbols = []

    def add_symbol(self, id, bp=0, symbol_name='Symbol',
                   symbol_base=SymbolBase):
        symbol_cls = self.symbol_table.get(id)
        if not symbol_cls:
            self.symbol_table[id] = symbol_cls = type(
                symbol_name,
                (symbol_base, ),
                dict(
                    id=id,
                    lbp=bp,
                    nud=lambda x: x,
                    first=None,
                    second=None
                )
            )
        symbol_cls.lbp = max([symbol_cls.lbp, bp])
        return symbol_cls

    def add_infix_op(self, id, bp=0):
        symbol_cls = self.add_symbol(id, bp)
        def led(self, left):
            self.first = left
            self.second = self.parser.parse_expression(bp)
            return self
        symbol_cls.led = led
        return symbol_cls

    def add_infix_r_op(self, id, bp=0):
        symbol_cls = self.add_symbol(id, bp)
        def led(self, left):
            self.first = left
            self.second = self.parser.parse_expression(bp - 1)
            return self
        symbol_cls.led = led
        return symbol_cls

    def add_prefix_op(self, id, bp=0):
        symbol_cls = self.add_symbol(id, bp)
        def nud(self):
            self.first = self.parser.parse_expression(bp)
            return self
        symbol_cls.nud = nud
        return symbol_cls

    def add_statement(self, id):
        symbol_cls = self.add_symbol(
            id,
            symbol_name='Statement',
            symbol_base=StatementBase
        )
        def std(self):
            self.parser.expect_symbol('indent')
            self.body = self.parser.parse_statements()
            return self
        symbol_cls.std = std
        return symbol_cls

    def symbol_for_token(self, token):
        try:
            symbol_cls = self.symbol_table[token.id]
        except KeyError:
            raise ParserError(
                token.lineno,
                'Unknown token {0}'.format(token.id)
            )
        return symbol_cls(self, token.value, token.lineno)

    def next_symbol(self):
        print 'next symbol'
        token = self._tokenizer.next()
        print token
        sym = self.symbol_for_token(token)
        print sym.id, repr(sym.value), sym.lineno
        self.symbols.append(sym)
        print 'next symbol end'
        return sym

    def peek_symbol(self):
        return self.symbol_for_token(self._tokenizer.peek())

    def expect_symbol(self, id):
        if self.symbol.id != id:
            raise ParserError(
                self.symbol.lineno,
                'Expected {0}, got {1}'.format(id, self.symbol.id)
            )
        try:
            self.symbol = self.next_symbol()
        except StopIteration:
            pass

    def expect_symbols(self, ids):
        for id in ids:
            try:
                self.expect_symbol(id)
            except ParserError as err:
                if not err.args[0].startswith('Expected'):
                    raise
            else:
                return
        last_id = ids.pop()
        raise ParserError(
            self.symbol.lineno,
            'Expected {0}, got {1}'.format(
                ', '.join(map(repr, ids)) + ' or ' + repr(last_id),
                repr(self.symbol.id)
            )
        )

    def parse_expression(self, rbp=0):
        print 'parse expression', rbp
        print self.symbol.id, self.symbol.value, self.symbol.lbp
        left_symbol, self.symbol = self.symbol, self.next_symbol()
        print self.symbol.id, self.symbol.value, self.symbol.lbp
        print 'null denotation'
        left_denotation = left_symbol.nud()
        print 'null denotation end'
        while rbp < self.symbol.lbp:
            left_symbol, self.symbol = self.symbol, self.next_symbol()
            print self.symbol.id, self.symbol.value, self.symbol.lbp
            print 'left denotation'
            left_denotation = left_symbol.led(left_denotation)
            print 'left denotation end'
        print 'parse expression end'
        return left_denotation

    def parse_statement(self):
        print 'parse statement'
        print self.symbol.id, self.symbol.value
        statement_symbol = self.symbol
        self.symbol = self.next_symbol()
        print self.symbol.id, self.symbol.value
        print 'statement denotation'
        result = statement_symbol.std()
        print 'statement denotation end'
        print 'parse statement end'
        return result

    def parse_statements(self):
        print 'parse statements'
        result = []
        while self.symbol.id not in ['end-statement', 'end']:
            print self.symbol.id, self.symbol.value
            if hasattr(self.symbol, 'std'):
                result.append(self.parse_statement())
            else:
                result.append(self.parse_expression())
            #try:
            #    self.symbol = self.next_symbol()
            #except StopIteration:
            #    return result
        print 'parse statements end'
        return result

    def parse(self, tokenizer):
        print 'parsing'
        self._tokenizer = tokenizer
        self.symbol = self.next_symbol()
        root = Root(self, Root.id, 0)
        root.body = self.parse_statements()
        del self._tokenizer
        print 'parsing end'
        return root

class Tokenizer(object):
    @classmethod
    def from_file(cls, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return cls(f.read())

    def __init__(self, source):
        self.line_iter = LineIterator(source)

        self.remaining = deque()
        self.in_block = False
        self.indent_length = None
        self.indent_level = 0
        self.lineno = 0
        self.exhausted = False

    def __iter__(self):
        return self

    def next_line(self):
        return self.line_iter.next()

    def next(self):
        if self.exhausted:
            raise StopIteration()
        if self.remaining:
            return self.remaining.popleft()
        try:
            lineno, line = self.next_line()
            self.lineno = lineno
        except StopIteration:
            self.exhausted = True
            return Token('end', self.lineno)
        line = self.handle_indent(lineno, line)
        if line[-1] == '>':
            self.handle_with_statement(lineno, line)
        elif line[-1] == ':':
            self.handle_block_statement(lineno, line)
        else:
            self.handle_expression(lineno, line)
        return self.next()

    def peek(self):
        next_token = self.next()
        self.remaining.appendleft(next_token)
        return next_token

    def handle_indent(self, lineno, line):
        indent, rest = indent_re.match(line).groups()
        if indent and not self.in_block:
            raise ParserError(lineno, 'unexpected indent')
        elif indent or self.in_block:
            indent_length = len(indent)
            if self.indent_length is None:
                self.indent_length = indent_length
                self.indent_level = 1
                self.remaining.append(Token('indent', lineno))
            elif indent_length != 0 and \
                gcd(indent_length, self.indent_length) != self.indent_length:
                raise ParserError(
                    lineno,
                    'indent must be a multiple of {0}'.format(
                        self.indent_length
                    )
                )
            elif indent_length != self.indent_length * self.indent_level:
                new_indent_level = indent_length / self.indent_length
                if new_indent_level < self.indent_level:
                    for _ in xrange(self.indent_level - new_indent_level):
                        self.remaining.append(Token('end-statement', lineno))
                elif new_indent_level - self.indent_level != 1:
                    raise ParserError(
                        lineno,
                        'expected 1 new level of indentation, got {0}'.format(
                            repr(new_indent_level - self.indent_level)
                        )
                    )
                else:
                    self.remaining.append(Token('indent', lineno))
                self.indent_level = new_indent_level
        return rest

    def handle_with_statement(self, lineno, line):
        identifier_prefix = line[:-1]
        self.remaining.append(Token('with', lineno, identifier_prefix))
        self.in_block = True

    def handle_block_statement(self, lineno, line):
        selector = line[:-1]
        self.remaining.append(Token('block', lineno, selector))
        self.in_block = True

    def handle_expression(self, lineno, line):
        rules = [
            (operator_re, lambda v: Token('list' if v == ',' else v, lineno)),
            (color_re, lambda v: Token('color', lineno, v)),
            (value_re, lambda v: Token('value', lineno, v)),
            (number_re, lambda v: Token('number', lineno, v)),
            (name_re, lambda v: Token('name', lineno, v)),
            (string_re, lambda v: Token(
                'string',
                lineno,
                v.strip('"').strip("'"))
            ),
        ]
        while line.strip():
            for expression_re, processor in rules:
                match = expression_re.match(line)
                if match is not None:
                    try:
                        value = match.group(1)
                    except IndexError:
                        value = None
                    self.remaining.append(processor(value))
                    line = line[match.end():].strip()
                    break
            else:
                raise ParserError(lineno, 'Syntax error')

nss_parser = Parser()
nss_parser.add_symbol('color')
nss_parser.add_symbol('value')
nss_parser.add_symbol('number')
nss_parser.add_symbol('name')
nss_parser.add_symbol('string')
nss_parser.add_symbol('indent').nud = lambda x: x
nss_parser.add_symbol('end-statement').nud = lambda x: x
nss_parser.add_symbol('end').nud = lambda x: x
nss_parser.add_symbol(')')

nss_parser.add_infix_op('=', 10)
nss_parser.add_infix_op(':', 10)
nss_parser.add_infix_op('list', 20)
nss_parser.add_infix_op('+', 30)
nss_parser.add_infix_op('-', 30)
nss_parser.add_infix_op('*', 40)
nss_parser.add_infix_op('/', 40)
nss_parser.add_infix_r_op('**', 50)
nss_parser.add_prefix_op('+', 60)
nss_parser.add_prefix_op('-', 60)
nss_parser.add_infix_op('.', 70)
nss_parser.add_prefix_op('(', 80)
nss_parser.add_statement('block')
nss_parser.add_statement('with')

def set(id, name):
    def decorate(func):
        setattr(nss_parser.symbol_table[id], name, func)
        return func
    return decorate

@set('list', 'led')
def led(self, left):
    self.body = []
    if left.id == 'list':
        self.body.extend(left.body)
    else:
        self.body.append(left)
    self.body.append(self.parser.parse_expression(self.lbp))
    return self

@set('(', 'nud')
def nud(self):
    self.first = self.parser.parse_expression(10)
    self.parser.expect_symbol(')')
    return self

@set('(', 'led')
def led(self, left):
    self.first = left
    self.second = self.parser.parse_expression(10)
    return self

@set('.', 'led')
def led(self, left):
    self.first = left
    self.second = self.parse_expression(self.lbp)
    if self.second.id != 'name':
        raise ParserError(
            self.lineno,
            'Expected name, got {0}'.format(self.second.id)
        )
    return self

def parse_nss(source):
    return nss_parser.parse(Tokenizer(source))
