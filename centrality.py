import math


def centrality(game, move):
    x, y = move
    gw, gh = game
    cx, cy = (math.ceil(gw / 2), math.ceil(gh / 2))
    return (x-cx)**2 + (y-cy)**2
    #return (gw - cx) ** 2 + (gh - cy) ** 2 - (x - cx) ** 2 - (y - cy) ** 2
    #return math.sqrt(math.fabs((gw - cx) ** 2 + (gh - cy) ** 2 - (x - cx) ** 2 - (y - cy) ** 2))


g = (7, 7)

row = range(0, 7)
col = range(0, 7)

for m in [(r, c) for r in row for c in col]:
    print(m, ' : ', centrality(g, m))
