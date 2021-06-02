# Partially from https://worksheets.codalab.org/worksheets/0xad3fc9f52f514e849b282a105b1e3f02

from data.alchemy.alchemy_artificial_generator import execute
from data.alchemy.utils import (
    check_well_formedness, word_to_int, colors, d_colors, int_to_word,
)

def find_nth_color(nth, color, objs):
    count = 0
    for pos, obj in enumerate(objs):
        if obj is None:
            # empty beaker (ignore in count)
            continue
        if color in obj:
            if nth == count: 
                return pos
            count += 1
    assert False, repr(color) + str(nth)

def countColor(color, objs):
    count = 0
    for obj in objs:
        if obj is None:
            # empty beaker (ignore in count)
            continue
        if color in obj:
            count += 1
    return count

def decode_pos(lst, objs):
    if len(lst) == 1:
        return word_to_int[lst[0]]
    elif all(c not in lst for c in colors):
        return word_to_int[" ".join(lst)]

    elif len(lst) >= 2: #TODO "second to last"

        if len(lst) == 2:
            nth_word, color = lst 
        else:
            color = lst[-1]
            nth_word = " ".join(lst[:-1])
        word_to_nth = {"first": 0,
                "second":1,
                "third":2,
                "fourth":3,
                "fifth":4,
                "second to last": countColor(color, objs) - 2,
                "last": countColor(color, objs) - 1,
                "": 0}
        #print(repr(nth_word))
        nth = word_to_nth[nth_word]
        return find_nth_color(nth, color, objs)


def parse_utt_with_world(string, world):
    objs = world['objects']

    lst = string.split(" ")
    action = lst[0]
    if action == 'drain':
        neg_amt = int(lst[1])
        pos = decode_pos(lst[4:-1], objs)

        if objs[pos] is None: cur_amt = 0
        else: cur_amt = len(objs[pos])
        amt = cur_amt - neg_amt 
        args = [pos, amt]
    elif action == 'pour':

        toBeakerIdx = lst.index('beaker')
        toP = decode_pos(lst[2:toBeakerIdx], objs)
        fromP = decode_pos(lst[toBeakerIdx+3:-1], objs)
        args = [toP, fromP]
    elif action == 'mix':
        arg = int(lst[-1])
        args = [arg]

    else: assert False, f'action: {action}, string: {string}'
    return action, args


def parse_world(string):
    #"r r oooo r pppp pp yyy"
    objs = []
    for beaker in string.split(' '):
        if beaker == '_': objs.append(None)
        else: objs.append([d_colors[c] for c in beaker])
    world = {'objects': objs, 'last': []}
    return world


def consistencyCheck(prior, gen):
    #parse prior:
    idx = prior.index('.')
    init_state = prior[:idx]
    prior_utts = prior[idx+2:]
    if len(prior_utts) > 0:
        if '\n' in prior_utts:
            prior_utts = prior_utts.split('.\n')
        else:
            prior_utts = prior_utts.split('. ')
    else: prior_utts = []
    init_state = " ".join(i[2:] for i in init_state.split(" "))
    gen_utt = gen[:-1]

    #run prior utterances
    world = parse_world(init_state)
    for u in prior_utts:
        action, args = parse_utt_with_world(u, world)
        world = execute(world, action, args) #except args need to be modified i guess

    #now do gen
    try:
        action, args = parse_utt_with_world(gen_utt, world)
    except (AssertionError, ValueError, KeyError) as e:
        return False
    try:
        new_world = execute(world, action, args) #except args need to be modified i guess
    except (TypeError, AssertionError) as e:
        return False

    if new_world['objects'] != world['objects']:
        return True
    return False


if __name__=='__main__':

    us = """142,drain 1 from the first red beaker. drain 1 from the third beaker. drain 1 from the third beaker. drain 2 from the last orange beaker. pour the third beaker to the last beaker. 
143,drain 1 from the last beaker. drain 2 from the first beaker. drain 2 from the fourth beaker. pour the last orange beaker to the  yellow beaker. mix beaker 6. 
144,drain 1 from the third beaker. drain 2 from the first beaker. drain 1 from the  orange beaker. pour the first red beaker to the  orange beaker. mix beaker 5. 
145,drain 1 from the first beaker. drain 1 from the first beaker. drain 1 from the fifth beaker. pour the first beaker to the fourth beaker. mix beaker 3. 
146,drain 1 from the first beaker. drain 2 from the fourth beaker. drain 1 from the second beaker. drain 3 from the third beaker. pour the first green beaker to the third beaker. 
147,drain 1 from the first beaker. drain 1 from the fourth beaker. drain 3 from the  purple beaker. drain 2 from the second beaker. pour the last green beaker to the  purple beaker. 
148,drain 2 from the fourth beaker. drain 2 from the first beaker. drain 2 from the fifth beaker. drain 1 from the last green beaker. pour the fourth beaker to the last green beaker. 
149,drain 1 from the fifth beaker. drain 1 from the first purple beaker. drain 3 from the third beaker. pour the fifth beaker to the third beaker. mix beaker 2. 
150,drain 2 from the fourth beaker. drain 2 from the second beaker. drain 2 from the second to last purple beaker. pour the fourth beaker to the second to last beaker. mix beaker 5. 
151,drain 1 from the first green beaker. drain 1 from the fourth beaker. drain 1 from the  purple beaker. drain 2 from the second green beaker. drain 1 from the last beaker. """

    states = """142,rrrr oooo ppp rr pp y oooo,rrr oooo ppp rr pp y oooo,rrr oooo pp rr pp y oooo,rrr oooo p rr pp y oooo,rrr oooo p rr pp y oo,rrr oooo _ rr pp y oop
143,ooo pppp oooo oooo ppp rrr yy,ooo pppp oooo oooo ppp rrr y,o pppp oooo oooo ppp rrr y,o pppp oooo oo ppp rrr y,o pppp oooo _ ppp rrr yoo,o pppp oooo _ ppp rrr bbb
144,ggg pppp rr ppp rr oo ggg,ggg pppp r ppp rr oo ggg,g pppp r ppp rr oo ggg,g pppp r ppp rr o ggg,g pppp _ ppp rr or ggg,g pppp _ ppp rr bb ggg
145,ppp ggg pppp r rrr pppp rrrr,pp ggg pppp r rrr pppp rrrr,p ggg pppp r rrr pppp rrrr,p ggg pppp r rr pppp rrrr,_ ggg pppp rp rr pppp rrrr,_ ggg pppp bb rr pppp rrrr
146,ggg oooo gggg oooo rr oooo yyyy,gg oooo gggg oooo rr oooo yyyy,gg oooo gggg oo rr oooo yyyy,gg ooo gggg oo rr oooo yyyy,gg ooo g oo rr oooo yyyy,_ ooo ggg oo rr oooo yyyy
147,ggg rrr yyyy ooo pppp ooo gg,gg rrr yyyy ooo pppp ooo gg,gg rrr yyyy oo pppp ooo gg,gg rrr yyyy oo p ooo gg,gg r yyyy oo p ooo gg,gg r yyyy oo pgg ooo _
148,ooo oooo gg ppp gggg rrrr r,ooo oooo gg p gggg rrrr r,o oooo gg p gggg rrrr r,o oooo gg p gg rrrr r,o oooo gg p g rrrr r,o oooo gg _ gp rrrr r
149,yyy pppp rrrr rrrr oo rrrr pppp,yyy pppp rrrr rrrr o rrrr pppp,yyy ppp rrrr rrrr o rrrr pppp,yyy ppp r rrrr o rrrr pppp,yyy ppp ro rrrr _ rrrr pppp,yyy ppp bb rrrr _ rrrr pppp
150,oo pppp yyyy yyy p pppp pppp,oo pppp yyyy y p pppp pppp,oo pp yyyy y p pppp pppp,oo pp yyyy y p pp pppp,oo pp yyyy _ p ppy pppp,oo pp yyyy _ p bbb pppp
151,ggg gggg pp ooo oooo ggg rrrr,gg gggg pp ooo oooo ggg rrrr,gg gggg pp oo oooo ggg rrrr,gg gggg p oo oooo ggg rrrr,gg gg p oo oooo ggg rrrr,gg gg p oo oooo ggg rrr
"""
    u = "drain 1 from the first red beaker"
    s0 = "ggg ggg rr pppp rrrr ggg ooo"
    s1 = "ggg ggg r pppp rrrr ggg ooo"


    for j in range(10):
        Us = [x for x in  us.split('\n')[j][4:].split(". ") if x != '']
        print(Us)

        Ss = states.split('\n')[j][4:].split(',')
        print(Ss)

        for i in range(5):


            s0 = Ss[i]
            s1 = Ss[i+1]
            u = Us[i]

            world = parse_world(s0)
            action, args = parse_utt_with_world(u, world)
            new_world = execute(world, action, args) #except args need to be modified i guess 

            assert new_world['objects'] == parse_world(s1)['objects']

            print("worked", j, i)


    print()
    print()
    prior ="""1:gggg 2:ggg 3:ggg 4:pppp 5:ppp 6:p 7:gggg. drain 1 from the fourth beaker.
drain 2 from the first beaker.
drain 2 from the second to last green beaker.
pour the last purple beaker to the second to last green beaker"""
    gt = "mix beaker 2."
    gen = "mix beaker 5."

    print(consistencyCheck(prior, gt))
    print()
    print()
    print()
    print(consistencyCheck(prior, gen))

