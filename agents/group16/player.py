import copy
import random

from Sequence.sequence_model import COORDS, SequenceState
from Sequence.sequence_utils import TRADSEQ, EMPTY, HOTBSEQ, JOKER, MULTSEQ
from template import Agent

#doublej,singlej,heart,seq_num,chip_num
draft_weight=[200,60,5,80,20]
#heart,selfseq,selfchips,oppseq,oppchip,using two-eyed jack
#using trained values
place_weight=[99.81,288.83,78.78,194.57,50.55,-69.61]
#heart,selfseq,selfchips,oppchips
#using trained values
remove_weight=[68.65,18.43,36.11,64.32]


class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.lastfeatures={'type':'empty','lastscore':0,'last_opscore':0,'last_q':0,'lastfeature':[]}
    def SelectAction(self, actions, game_state:SequenceState):
        action=actions[0]
        #get basic information of the game state
        clr, sclr = game_state.agents[self.id].colour, game_state.agents[self.id].seq_colour
        oc, os = game_state.agents[self.id].opp_colour, game_state.agents[self.id].opp_seq_colour
        draft=game_state.board.draft
        chips=game_state.board.chips
        hand=game_state.agents[self.id].hand


        #find bestdraft
        draftchips = copy.deepcopy(chips)
        for card in hand:
            if card not in ['jd','jc','jh','js']:
                coords=COORDS[card]
                for x,y in coords:
                    if draftchips[x][y]==EMPTY:
                        draftchips[x][y]=clr

        BestQValue_draft=0
        Best_draft=draft[0]
        for card in draft:
            draft_features=self.CalculateDraftFeatures(card,draftchips,(clr,sclr,oc,os))
            q_value=self.Q_VALUE(draft_features,draft_weight)
            if q_value>BestQValue_draft:
                BestQValue_draft=q_value
                Best_draft=card
        betterdraft_actionList=[]
        #when it is turn of 'trade',choose a best draft and trade
        if action['type']=='trade':
            for a in actions:
                if a['draft_card']==Best_draft:
                    return a
            return random.choice(actions)
        else:
            for a in actions:
                if a['draft_card']==Best_draft:
                    betterdraft_actionList.append(a)
        BestAction=betterdraft_actionList[0]
        BestQValue = 0
        # when it is not turn of 'trade',choose the best action with both best coordinate of remove/place
        # and best draft
        for a in betterdraft_actionList:
            if a['type']=='place':
                place_features = self.CalculatePlaceFeatures(a['coords'], chips, (clr, sclr, oc, os))
                if a['play_card'] in ['jd','jc']:
                    place_features.append(0.1)
                else:
                    place_features.append(0)
                q_value = self.Q_VALUE(place_features, place_weight)
                if q_value>BestQValue:
                    BestQValue = q_value
                    BestAction = a
            else:
                remove_features = self.CalculateRemoveFeatures(a['coords'], chips, (clr, sclr, oc, os))
                q_value = self.Q_VALUE(remove_features, remove_weight)
                for card in hand:
                    coords=COORDS[card]
                    for coord in coords:
                        if a['coords']==coord:
                            q_value*=1.13
                if q_value > BestQValue:
                    BestQValue = q_value
                    BestAction = a

        return BestAction

    def Q_VALUE(self,features,weights):
        value=0
        if len(features)==len(weights):
            for i in range(len(weights)):
                value+=features[i]*weights[i]
        else:
            print('features do not match weights!')
        return value

    # generate draft features' value
    def CalculateDraftFeatures(self, card, chips, colortuple):
        if card in ['jd', 'jc']:
            return [1, 0, 0, 0, 0]
        if card in ['jh', 'js']:
            return [0, 1, 0, 0, 0]
        draftFeatures = [0, 0, 0, 0, 0]
        clr, sclr, oc, os = colortuple
        if card in ['2h', '3h', '4h', '5h']:
            draftFeatures[2] = 1
        #             hcoords = [(4, 4), (4, 5), (5, 4), (5, 5)]
        #             for coord in hcoords:
        #                 if coord == oc or coord == os:
        #                     draftFeatures[2] = 0
        #                     break
        coords = COORDS[card]
        max_s = 0
        max_c = 0
        for x, y in coords:
            if chips[x][y] == EMPTY:
                chips[x][y] = clr
                num_seq, num_chips = self.Seqrewards(chips, colortuple, (x, y))
                if num_seq > max_s:
                    max_s = num_seq
                    max_c = num_chips
                elif num_seq == max_s and num_chips > max_c:
                    max_s = num_seq
                    max_c = num_chips
                chips[x][y] = EMPTY
        draftFeatures[3] = max_s
        draftFeatures[4] = max_c
        return draftFeatures

    # generate place features' value
    def CalculatePlaceFeatures(self,coord,chips,colortuple):
        clr, sclr, oc, os = colortuple
        features=[0,0,0,0,0]
        #can optimize
        if coord in [(4, 4), (4, 5), (5, 4), (5, 5)]:
            features[0]=0.1
        x,y=coord
        chips[x][y]=clr
        seq_num,chips_num=self.Seqrewards(chips,colortuple,coord)
        features[1]=seq_num
        features[2]=chips_num
        chips[x][y] = oc
        seq_num,chips_num=self.Seqrewards(chips, (oc, os,clr, sclr),coord)
        features[3]=seq_num
        features[4]=chips_num
        chips[x][y]=EMPTY
        return features

    # generate remove features' value
    def CalculateRemoveFeatures(self,coord,chips,colortuple):
        clr, sclr, oc, os = colortuple
        features=[0,0,0,0]
        #can optimize
        if coord in [(4, 4), (4, 5), (5, 4), (5, 5)]:
            features[0]=0.1
        x,y=coord
        chips[x][y]=clr
        seq_num,chips_num=self.Seqrewards(chips,colortuple,coord)
        features[1]=seq_num
        features[2]=chips_num
        chips[x][y] = oc
        seq_num,chips_num=self.Seqrewards(chips, (oc, os,clr, sclr),coord)
        features[3]=chips_num
        chips[x][y]=EMPTY
        return features

    # calculate rewards if allied chips are at the same diagonal/line/row of last_coords
    # mostly copy from def checkSeq(self, chips, plr_state, last_coords) in sequence_model.py
    def Seqrewards(self, chips, colortuple, last_coords):
        clr, sclr, oc, os = colortuple
        "the beginning of copied code part 1 from checkSeq(self, chips, plr_state, last_coords)"
        seq_type = TRADSEQ
        seq_coords = []
        seq_found = {'vr': 0, 'hz': 0, 'd1': 0, 'd2': 0, 'hb': 0}
        found = False
        nine_chip = lambda x, clr: len(x) == 9 and len(set(x)) == 1 and clr in x
        lr, lc = last_coords
        # All joker spaces become player chips for the purposes of sequence checking.
        for r, c in COORDS['jk']:
            chips[r][c] = clr

        # First, check "heart of the board" (2h, 3h, 4h, 5h). If possessed by one team, the game is over.
        coord_list = [(4, 4), (4, 5), (5, 4), (5, 5)]
        heart_chips = [chips[y][x] for x, y in coord_list]
        if EMPTY not in heart_chips and (clr in heart_chips or sclr in heart_chips) and not (
                oc in heart_chips or os in heart_chips):
            seq_type = HOTBSEQ
            seq_found['hb'] += 2
            seq_coords.append(coord_list)

        # Search vertical, horizontal, and both diagonals.
        vr = [(-4, 0), (-3, 0), (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
        hz = [(0, -4), (0, -3), (0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
        d1 = [(-4, -4), (-3, -3), (-2, -2), (-1, -1), (0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
        d2 = [(-4, 4), (-3, 3), (-2, 2), (-1, 1), (0, 0), (1, -1), (2, -2), (3, -3), (4, -4)]
        for seq, seq_name in [(vr, 'vr'), (hz, 'hz'), (d1, 'd1'), (d2, 'd2')]:
            coord_list = [(r + lr, c + lc) for r, c in seq]
            coord_list = [i for i in coord_list if 0 <= min(i) and 9 >= max(i)]  # Sequences must stay on the board.
            chip_str = ''.join([chips[r][c] for r, c in coord_list])
            "end of copy part1"
            #check allied chips at the same column/row/diagonal
            cont = 0
            cont_re = 0
            room = 0
            sclr_check = False
            chip_rewards = 0
            for (x1, y1) in vr:
                if lr + x1 < 10 and lr + x1 >= 0 and (chips[lr + x1][lc] == clr or
                                                      (chips[lr + x1][lc] == sclr and sclr_check == False)):
                    if chips[lr + x1][lc] == sclr:
                        sclr_check = True
                    # add human experience of checking future development
                    if lr + x1 > 0 and chips[lr + x1 - 1][lc] == EMPTY:
                        room += 1
                    elif lr + x1 < 9 and chips[lr + x1 + 1][lc] == EMPTY:
                        room += 1
                    cont += 1
                    cont_re += 1 + 0.2 * (cont - 1)
                else:
                    chip_rewards += cont_re * room
                    cont_re = 0
                    cont = 0
                    room = 0
            cont = 0
            cont_re = 0
            room = 0
            sclr_check = False
            for (x1, y1) in hz:
                if lc + y1 < 10 and lc + y1 >= 0 and (chips[lr][lc + y1] == clr or
                                                      (chips[lr][lc + y1] == sclr and sclr_check == False)):
                    if chips[lr][lc + y1] == sclr:
                        sclr_check = True
                    # add human experience of checking future development
                    if lc + y1 > 0 and chips[lr][lc + y1 - 1] == EMPTY:
                        room += 1
                    elif lc + y1 < 9 and chips[lr][lc + y1 + 1] == EMPTY:
                        room += 1
                    cont += 1
                    cont_re += 1 + 0.2 * (cont - 1)
                else:
                    chip_rewards += cont_re * room
                    cont_re = 0
                    cont = 0
                    room = 0
            cont = 0
            cont_re = 0
            room = 0
            sclr_check = False
            for (x1, y1) in d1:
                if lr + x1 < 10 and lr + x1 >= 0 and lc + y1 < 10 and lc + y1 >= 0 and \
                        (chips[lr + x1][lc + y1] == clr or (chips[lr + x1][lc + y1] == sclr and sclr_check == False)):
                    if chips[lr + x1][lc + y1] == sclr:
                        sclr_check = True
                    # add human experience of checking future development
                    if lc + y1 > 0 and lr + x1 > 0 and chips[lr + x1 - 1][lc + y1 - 1] == EMPTY:
                        room += 1
                    elif lc + y1 < 9 and lr + x1 < 9 and chips[lr + x1 + 1][lc + y1 + 1] == EMPTY:
                        room += 1
                    cont += 1
                    cont_re += 1 + 0.2 * (cont - 1)
                else:
                    chip_rewards += cont_re * room
                    cont_re = 0
                    cont = 0
                    room = 0
            cont = 0
            cont_re = 0
            room = 0
            sclr_check = False
            for (x1, y1) in d2:
                if lr + x1 < 10 and lr + x1 >= 0 and lc + y1 < 10 and lc + y1 >= 0 and \
                        (chips[lr + x1][lc + y1] == clr or (chips[lr + x1][lc + y1] == sclr and sclr_check == False)):
                    if chips[lr + x1][lc + y1] == sclr:
                        sclr_check = True
                    # add human experience of checking future development
                    if lc + y1 > 0 and lr + x1 < 9 and chips[lr + x1 + 1][lc + y1 - 1] == EMPTY:
                        room += 1
                    elif lc + y1 < 9 and lr + x1 > 0 and chips[lr + x1 - 1][lc + y1 + 1] == EMPTY:
                        room += 1
                    cont += 1
                    cont_re += 1 + 0.2 * (cont - 1)
                else:
                    chip_rewards += cont_re * room
                    cont_re = 0
                    cont = 0
                    room = 0

            "beginning of copied code part 2"
            # Check if there exists 4 player chips either side of new chip (counts as forming 2 sequences).
            if nine_chip(chip_str, clr):
                seq_found[seq_name] += 2
                seq_coords.append(coord_list)
            # If this potential sequence doesn't overlap an established sequence, do fast check.
            if sclr not in chip_str:
                sequence_len = 0
                start_idx = 0
                for i in range(len(chip_str)):
                    if chip_str[i] == clr:
                        sequence_len += 1
                    else:
                        start_idx = i + 1
                        sequence_len = 0
                    if sequence_len >= 5:
                        seq_found[seq_name] += 1
                        seq_coords.append(coord_list[start_idx:start_idx + 5])
                        break
            else:  # Check for sequences of 5 player chips, with a max. 1 chip from an existing sequence.
                for pattern in [clr * 5, clr * 4 + sclr, clr * 3 + sclr + clr, clr * 2 + sclr + clr * 2,
                                clr + sclr + clr * 3, sclr + clr * 4]:
                    for start_idx in range(5):
                        if chip_str[start_idx:start_idx + 5] == pattern:
                            seq_found[seq_name] += 1
                            seq_coords.append(coord_list[start_idx:start_idx + 5])
                            found = True
                            break
                    if found:
                        break

        for r, c in COORDS['jk']:
            chips[r][c] = JOKER  # Joker spaces reset after sequence checking.

        num_seq = sum(seq_found.values())
        if num_seq > 1 and seq_type != HOTBSEQ:
            seq_type = MULTSEQ
        "end of copied code part2"
        return num_seq, float(chip_rewards / 20)


