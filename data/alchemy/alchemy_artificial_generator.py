# from https://worksheets.codalab.org/worksheets/0xad3fc9f52f514e849b282a105b1e3f02

import random
import os
import csv
import argparse
import copy

EMPTY_CHAR = "_"
MIX_COLOR = "brown"


def render(world):
    objs = world['objects']
    out = []

    padding = 2.5
    liquid_padding = 1.
    beaker_padding = 25 
    torso_width = 50 
    torso_height = torso_width  
    world_width = len(objs) * (torso_width + 2 * padding + beaker_padding)
    world_height = torso_height + 15 * padding
    line_length = 5 

    out.append('<svg width="%d" height="%d">' % (world_width, world_height))

    beaker_y_bot = world_height - 10 - torso_height 
    beaker_height = 8
    beaker_sharpness = 5 


    for i, obj in enumerate(objs):
        x = beaker_padding / 2 + i * (torso_width + 2 * padding + beaker_padding)
        y = world_height - 10
        out.append('  <rect x="%d" y="%d" width="%d" height="%d" style="fill:%s;stroke-width:1;stroke:rgb(0,0,0)"/>' % \
                (x + padding, y - torso_height, torso_width, torso_height, 'white'))

        y = world_height - 10
        if obj is not None:
            for color in obj:
                out.append('  <rect x="%d" y="%d" width="%d" height="%d" style="fill:%s;stroke-width:0;stroke:rgb(0,0,0)"/>' % \
                    (x + padding + liquid_padding, y - torso_height / 4, torso_width - 2 * liquid_padding, torso_height / 4, color))
                y -= torso_height / 4
        y = world_height - 10 
        for _ in range(4):
            out.append(' <line x1="%d" y1="%d" x2="%d" y2="%d" style="stroke:rgb(0,0,0);stroke-width:1"/>' % \
                (x + padding, y, x + padding + line_length, y))
            y -= torso_height / 4
        beaker_x_start = x - beaker_sharpness 
        beaker_x_end = x + torso_width + padding
        out.append('<polygon points = "%d,%d %d,%d %d,%d %d,%d"style="fill:white;stroke:rgb(0,0,0);stroke-width:1" />' % \
            (beaker_x_start, y - beaker_height, x + 1.2 * padding, y, beaker_x_end, y, beaker_x_end, y - beaker_height, ))
        out.append(' <line x1="%d" y1="%d" x2="%d" y2="%d" style="stroke:white;stroke-width:2"/>' %  
            (x + padding + line_length, y, x + torso_width + liquid_padding, y))
    out.append('  <rect x="%d" y="%d" width="%d" height="%d" style="fill:%s;stroke-width:2;stroke:rgb(0,0,0)"/>' % \
            (0, world_height - 10, world_width, 10, 'brown'))
    out.append('</svg>')
    return '\n'.join(out)


def choice(l):
    l = [x for x in l if x != None]
    return None if l == [] else random.choice(l)

def get_object_positions(world):
    return [i for i, obj in enumerate(world['objects']) if obj != None]

def execute(world, action, args):
    # drain, add, stir
    objs = copy.deepcopy(world['objects'])
    last = None
    if action == 'drain':
        pos, amt = args
        spec = objs[pos]
        assert(spec is not None)
        last = pos
        # amount to keep
        objs[pos] = objs[pos][:amt]
        if objs[pos] == []:
            objs[pos] = None
    # check that you don't overflow
    elif action == 'pour':
        src, dest = args
        # spec = array
        if objs[dest] is None:
            objs[dest] = reversed(src)
        else:
            objs[dest] += reversed(objs[src])
        objs[src] = None
        last = dest
        assert(len(objs[dest]) <= 4)
    elif action == 'mix':
        pos, = args
        spec = objs[pos]
        assert(len(spec) != 1)
        last = pos
        assert(spec is not None)
        for i, x in enumerate(spec):
            spec[i] = MIX_COLOR 
        # set color to brown
    return {'objects': objs, 'last': last}

def can_mix(obj):
    if obj is None: return False
    colors = set()
    for color in obj:
        colors.add(color)
    return len(colors) > 1

def can_mix_in_world(objs):
    for i, x in enumerate(objs):
        if can_mix(x):
            return True
    return False

def get_positions(world):
    objs = world['objects']
    positions = []
    for i, x in enumerate(objs):
        if x is not None:
            positions.append(i)
    return positions

def nth_color(world, index):
    objs = world['objects']
    counter = 0
    color = objs[index][0]
    for i in range(index):
        beaker = objs[i]
        if objs[i] is not None and color in beaker:
            counter += 1
    return counter 

def remaining_color(world, index):
    objs = world['objects']
    counter = 0
    col = objs[index][0]
    for i in range(index+1, len(objs)):
        if objs[i] is not None and col in objs[i]:
            counter += 1
    return counter

def color(world, index):
    objs = world['objects']
    return objs[index][0]

def beaker_name(world, index):
    objs = world['objects']
    beaker_color = color(world, index)
    nth = nth_color(world, index)
    remaining = remaining_color(world, index)
    name =""
    # If beaker has just 1 color, with probability 0.5, use nth color
    if len(set(objs[index])) == 1 and random.random() > 0.5:
        if nth == 0 and remaining == 0:
            name = ""
        elif nth == 0:
            name = "first" 
        elif remaining == 0:
            name = "last"
        elif nth == 1:
            name = "second"
        elif remaining == 1:
            name = "second to last"
        elif nth == 2:
            name = "third"
        name += " " + beaker_color
    else:
        if index == 0:
            name = "first"
        elif index == len(objs) - 1:
            name = "last"
        elif index == 1:
            name = "second"
        elif index == 2:
            name = "third"
        elif index == 3:
            name = "fourth"
        elif index == 5:
            name = "second to last"
        elif index == 4:
            name = "fifth"
    return name

# give preference to adding to same thing
def pick_action(world):
    # mix last thing should be most likely
    objs = world['objects']
    last = world['last']
    action = None
    args = None
    u = ""
    while True:
        if action == None or random.random() < 0.01:
            if can_mix_in_world(objs):
                action = choice(['mix', 'pour', 'drain'])
            else:
                action = choice(['pour', 'drain'])
        
        try:
            # pick two indices
            if action == 'pour':
                src = choice(get_positions(world))
                if src == None: continue
                dest = None
                if last == []:
                    dest = choice(get_positions(world))
                else:
                    dest = choice([last, choice(get_positions(world))])
                if src == dest: continue
                if objs[dest] is None: continue
                if objs[src] is None: continue
                if len(objs[dest] + objs[src]) > 4: continue
                u = "pour the " + beaker_name(world, src) + " beaker to the " + beaker_name(world, dest) + " beaker"
                args = [src, dest]
            elif action == 'drain':
                pos = choice(get_positions(world))
                if pos == None: continue
                # amounts to keep
                possible_amts = [x for x in range(len(objs[pos]))]
                amt = choice(possible_amts)
                if amt is None: continue
                args = [pos, amt]
                u = "drain " + str(len(objs[pos]) - amt) + " from the " + beaker_name(world, pos) + " beaker" 
            elif action == 'mix':
                pos = choice([last, choice(get_positions(world))])
                if pos is None or pos == []: continue
                if not can_mix(objs[pos]): continue
                args = [pos]
                u = "mix beaker " + str(pos)
        except Exception as err:
            # should not happen
            print(f'Error {err}: ' + str([''.join([bentry[0] for bentry in beaker]) if beaker is not None else '' for beaker in world['objects']]) + ' ' + action)
            raise Exception
        break
    assert(args is not None)
    return (action, args, u)
        
def advance(world):
    objs = world['objects']
    last = world['last']
    empty_world = True
    for o in objs:
        if o is not None:
            empty_world = False
    if empty_world:
        raise Exception("No Actions Possible")
    action, args, u = pick_action(world)
    world = execute(world, action, args)
    return (world, u)

def init_objects(num_positions):
    objects = []
    for i in range(num_positions):
        amt = choice([0,1,2,3,4])
        color = choice(colors)
        beaker = None 
        for _ in range(amt):
            if beaker is None:
                beaker = []
            beaker.append(color)
        objects.append(beaker)
    return objects

def form_state(obj):
    if obj is None:
        return "_"
    else:
        string = ""
        for x in obj:
            string += x[0]
        return string 

def form_world(objects):
    values = []
    for obj in objects:
        values.append(form_state(obj))
    return " ".join(values)

# keep brown as a mixture only
colors = ['green', 'red', 'orange', 'purple', 'yellow']
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_positions', default=7)
    parser.add_argument('--num_actions', default=5)
    parser.add_argument('--num_scenarios', default=1)
    parser.add_argument('--output', default="tasks")
    args = parser.parse_args()

    id_file = open(os.getcwd() + "/id", "r+")
    cur_id = int(id_file.read())

    os.mkdir(args.output)
    os.chdir(args.output)
    f = open(args.output + ".csv", 'w')
    task_file = csv.writer(f)
    g = open(args.output + "_answers.csv", 'w')
    h = open("batch.csv", 'w')
    batch = csv.writer(h) 

    answer_file = csv.writer(g)
    if not os.path.exists(args.output + "_gen"):
        os.mkdir(args.output + "_gen")
    os.chdir(os.getcwd() + "/" + args.output + "_gen")

    headers = ['world_' + str(i) for i in range(int(args.num_actions)+1)]
    headers.insert(0, "Input_id")
    task_file.writerow(headers)
    answer_file.writerow(headers)
    batch_headers = ["Input.Input_id", "Answer.story"]
    batch.writerow(batch_headers)

    # make more robust by ending early if nothing left
    i = 0
    while i < int(args.num_scenarios):
        try:
            scenario = []
            scenario_answers = []
            objects = init_objects(int(args.num_positions))
            world = {'objects': objects, 'last': []}
            scenario.append(cur_id)
            scenario_answers.append(cur_id)
            scenario.append(render(world).replace('\n', ''))
            scenario_answers.append(form_world(world['objects']))
            out = open(os.getcwd() + '/gen' + str(cur_id) + '.html', 'w')
            print(render(world), file=out)
            print("<br>", file=out)
            utterance = ""
            for _ in range(int(args.num_actions)):
                world, u = advance(world)
                utterance += u + ". "
                scenario.append(render(world).replace('\n', ''))
                scenario_answers.append(form_world(world['objects']))
                print(render(world), file=out)
                print("<br>", file=out)
            task_file.writerow(scenario)
            answer_file.writerow(scenario_answers)
            batch.writerow([cur_id, utterance])
            cur_id += 1
            out.close()
            i += 1
        except Exception as err:
            continue

    f.close()
    g.close()
    id_file.seek(0)
    id_file.write(str(cur_id))
    id_file.truncate()
    id_file.close()
    h.close()