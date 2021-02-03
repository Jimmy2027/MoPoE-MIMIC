# HK, 11.01.21
# https://github.com/roger-/PyTikZ/blob/master/tikz.py
# pytikz v0.13
# A simple Python -> TikZ wrapper.
# See main() for sample usage.
#
# 2013, -Roger

# https://www.overleaf.com/learn/latex/LaTeX_Graphics_using_TikZ:_A_Tutorial_for_Beginners_(Part_3)%E2%80%94Creating_Flowcharts
# https://www.overleaf.com/learn/latex/TikZ_package
# https://github.com/negrinho/sane_tikz
from contextlib import contextmanager
from functools import wraps

__all__ = ['ra', 'Picture']

# change this to use spaces instead of tabs
TAB = '\t'
LINE_TYPES = {'straight': '--', 'hv': '-|', 'vh': '|-', 'curved': '..'}


def ra(radius, angle_deg):
    return '%s:%s' % (angle_deg, radius)


def format_pos(pos, coord='abs'):
    if isinstance(pos, complex):
        pos = (pos.real, pos.imag)
    elif isinstance(pos, (int, float)):
        pos = '(%s, 0)' % pos
    elif isinstance(pos, str):
        if pos != 'cycle':
            pos = '(%s)' % pos

    if coord in ('rel', 'relative', '+'):
        coord_type = '+'
    elif coord in ('inc', 'increment', 'incremental', '++'):
        coord_type = '++'
    else:
        coord_type = ''

    return coord_type + str(pos)


def to_command(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapped


class Path(object):
    def __init__(self, style=''):
        self.style = style

        c = '\\path[%s]' % (self.style)
        self.contents = [c]

    def add_command(self, cmd, to=False):
        # add command to new line if it changes the current path
        if to:
            self.contents.append('\t' + cmd)
        else:
            self.contents[-1] += '\t' + cmd

    def at(self, pos, name='', coord='abs'):
        c = format_pos(pos, coord)
        self.add_command(c, to=True)

        if name:
            self.coord(name=name)

        return self

    def coord(self, style='', name=''):
        c = 'coordinate[%s] (%s)' % (style, name)
        self.add_command(c, to=False)

        return self

    def line_to(self, pos, type='straight', name='', coord='abs'):
        c = '%s %s' % (LINE_TYPES[type], format_pos(pos, coord))
        self.add_command(c, to=True)

        if name:
            self.coord(name=name)

        return self

    def to(self, pos, style='', name='', coord='abs'):
        c = 'to[%s] %s' % (style, format_pos(pos, coord))
        self.add_command(c, to=True)

        if name:
            self.coord(name=name)

        return self

    def spline_to(self, pos, control1, control2=None, name='', coord='abs'):
        if not control2:
            control2 = control1

        c = '.. controls %s and %s .. %s' % (format_pos(control1, coord), \
                                             format_pos(control2, coord), \
                                             format_pos(pos, coord))
        self.add_command(c, to=True)

        if name:
            self.coord(name=name)

        return self

    def node(self, text='', style='', name=''):
        c = 'node[%s] (%s) {%s}' % (style, name, text)
        self.add_command(c, to=False)

        return self

    def circle(self, radius):
        c = 'circle (%s)' % radius
        self.add_command(c, to=False)

        return self

    def arc_to(self, params, coord='abs'):
        c = 'arc (%s)' % params
        self.add_command(c, to=True)

        return self

    def rect_to(self, pos, coord='abs'):
        c = 'rectangle %s' % format_pos(pos, coord)
        self.add_command(c, to=True)

        return self

    def grid_to(self, pos, coord='abs'):
        c = 'grid %s' % format_pos(pos, coord)
        self.add_command(c, to=True)

        return self

    def cycle(self):
        self.line_to('cycle')

    def make(self):
        # end current statement
        self.contents[-1] += ';'

        return self.contents


class Picture(object):
    def __init__(self, style=''):
        self.style = style
        self.contents = []

    def add_command(self, text):
        self.contents.append('\t' + text)

    @contextmanager
    def path(self, style='draw'):
        path = Path(style)
        yield path
        c = path.make()

        # fit short paths on one line
        if len(c) <= 3:
            c = [''.join(c).replace('\n', ' ')]

        for line in c:
            self.add_command(line)

    @contextmanager
    def new_scope(self, style=''):
        scope = Picture(self.style)
        yield scope
        c = scope.contents

        self.add_command('\\begin{scope}[%s]' % style)
        for line in c:
            self.add_command('\t' + line)
        self.add_command('\\end{scope}')

    def set_coord(self, pos, options='', name='', coord='abs'):
        cmd = '\\coordinate[%s] (%s) at %s;' % (options, name, format_pos(pos, coord))
        self.add_command(cmd)

    def set_node(self, pos=None, text='', options='', name='', coord='abs'):
        if pos:
            cmd = f'\\node[{options}] ({name}) at {format_pos(pos, coord)} {{{text}}};'
        else:
            cmd = f'\\node[{options}] ({name}) {{{text}}};'
        self.add_command(cmd)

    def add_node(self, text='', options='', name='', args: dict = None):
        cmd = f'\\node[{options}] ({name})'
        if args:
            args_str = ''
            if 'style' in args:
                args_str += f'{args["style"]}, '
            for k, v in args.items():
                if k != 'style':
                    args_str += f'{k}={v}, '
            cmd += f' [{args_str}]'
        cmd += f' {{{text}}};'
        self.add_command(cmd)

    def set_line(self, origin, destination, exit_direction=None, input_direction=None, line_type='straight',
                 label=None, label_pos='', edge_options=''):

        cmd = f'\\draw[->] ({origin}'
        if exit_direction:
            cmd += f'.{exit_direction}'
        if edge_options:
            cmd += f') edge[{edge_options}]'
        else:
            cmd += f') {LINE_TYPES[line_type]}'
        # need to create node for text on line
        if label:
            cmd += ' node'
            if label_pos:
                cmd += f'[anchor={label_pos}]'
            cmd += f' {{{label}}}'
        cmd += f' ({destination}'
        if input_direction:
            cmd += f'.{input_direction}'
        cmd += f');'
        self.add_command(cmd)

    def __setitem__(self, item, value):
        self.set_coord(pos=value, name=item)

    def make(self):
        c = '\\begin{tikzpicture}[%s]\n' % (self.style)
        c += '\n'.join(self.contents)
        c += '\n\\end{tikzpicture}'

        # use another character(s) instead of TAB
        c = c.replace('\t', TAB)

        return c


def main():
    pic = Picture()

    # define some internal TikZ coordinates
    pic['pointA'] = 5 + 0j  # complex number for x-y coordinates
    pic['pointB'] = '45:5'  # or string for any native TikZ format

    # define a coordinate in Python
    pointC = 3 + 1j

    # draw new path
    with pic.path('draw') as draw:
        draw.at(0 + 0j, name='start').grid_to(6 + 6j)

    with pic.path('fill, color=red') as draw:
        draw.at('start') \
            .line_to('pointA', coord='relative') \
            .line_to('pointB').node('hello', style='above, black') \
            .line_to(pointC) \
            .spline_to(5 + 1j, 2 + 5j).node('spline') \
            .cycle()  # close path

    with pic.path('fill, color=blue') as draw:
        draw.at(pointC).circle(0.3)

    print(pic.make())


if __name__ == '__main__':
    main()

