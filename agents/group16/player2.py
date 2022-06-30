
import copy
import random
from template import Agent
from Sequence.sequence_utils import *
from Sequence.sequence_model import SequenceGameRule, SequenceState, COORDS, BOARD

class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.mcts_obj = MCTS(self.id)

    def SelectAction(self, actions, game_state):
        simulator = TrainingSequenceGameRule(copy.deepcopy(game_state), self.id, 4)
        state = copy.deepcopy(simulator.current_game_state)
        root_node = self.mcts_obj.mcts(simulator, state, self.id)

        return root_node.best_action()


class StateNode:

    def __init__(self, parent, state, agent_id):
        self.parent = parent
        self.state = state
        self.children = {}
        self.agent_id = agent_id
        self.visits = 0
        self.value = 0
        self.reward = 0
        self.avrg = 0

    def get_value(self):
        return self.value

    def select(self, simulator, state):
        if not self.isFullyExpanded(simulator, state):
            return self
        else:
            actions = list(self.children.keys())
            q_values = []
            # 拿到所有子节点的value
            for action_tuple in actions:
                q_values.append(self.children[action_tuple].get_value())
            # e-greedy选择下一个节点
            random_number = random.random()
            if random_number < 0.01:
                selected_action = random.choice(actions)
                while self.children[selected_action].value == 0:
                    selected_action = random.choice(actions)
            else:
                max_q_value = -1
                best_actions = []
                for action_tuple, q_value in zip(actions, q_values):
                    if q_value > max_q_value:
                        max_q_value = q_value
                        best_actions = [action_tuple]
                    elif q_value == max_q_value:
                        best_actions.append(action_tuple)
                selected_action = random.choice(best_actions)
            child_node = self.children[selected_action]
            return child_node.select(simulator, child_node.state)

    def isFullyExpanded(self, simulator, state):
        valid_actions = simulator.getLegalActions(state, self.agent_id)
        valid_actions = set([self.dict2tuple(a) for a in valid_actions])

        if len(valid_actions) == len(self.children):
            return True
        else:
            return False

    def dict2tuple(self, dict):
        return tuple(dict.items())

    def tuple2dict(self, tuple):
        return dict(tuple)

    # 剪掉的是那些节点
    def expand(self, simulator, node):
        legal_actions = simulator.getLegalActions(node.state, self.agent_id)
        legal_actions_tuple = set([self.dict2tuple(a) for a in legal_actions])
        valid_actions = legal_actions_tuple - self.children.keys()

        # 选择子节点
        action = random.choice(list(valid_actions))
        action = self.tuple2dict(action)
        new_state, rwd = simulator.generateSuccessorWithReward(copy.deepcopy(node.state), action, self.agent_id, False)
        new_child = StateNode(self, new_state, self.agent_id)
        new_child.reward = rwd
        self.children[self.dict2tuple(action)] = new_child
        self.avrg = rwd

        while rwd < self.avrg:
            child_num = len(self.children.keys())
            sum_r = self.avrg * (child_num - 1) + rwd
            self.avrg = sum_r / child_num
            new_child.value = 0
            valid_actions = legal_actions_tuple - self.children.keys()
            action = random.choice(list(valid_actions))
            action = self.tuple2dict(action)
            new_state, rwd = simulator.generateSuccessorWithReward(copy.deepcopy(node.state), action, self.agent_id, False)
            new_child = StateNode(self, new_state, self.agent_id)
            new_child.reward = rwd
            self.children[self.dict2tuple(action)] = new_child

        return new_child

    def backPropagate(self, reward):
        self.visits += 1
        self.value = self.value + ((self.reward + reward - self.value) / self.visits)

        if self.parent != None:
            reward *= 0.9
            self.parent.backPropagate(reward)

    def best_action(self):
        q_values = {}
        for action in self.children.keys():
            q_values[(self.state, action)] = self.children[action].get_value()

        max_q = -1
        best_actions = []
        for (state, action), q_value in q_values.items():
            if q_value > max_q:
                max_q = q_value
                best_actions = [action]
            elif q_value == max_q:
                best_actions.append(action)
        return self.tuple2dict(random.choice(best_actions))

class MCTS:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.all_nodes = {}

    def mcts(self, simulator, state, agent_id):
        root_node = StateNode(None, state, agent_id)
        episode_done = 0
        while episode_done < 170:
            selected_node = root_node.select(simulator, state)

            if not simulator.endState(selected_node.state):
                child = selected_node.expand(simulator, selected_node)
                reward = self.simulate(simulator, child, self.agent_id)
                child.backPropagate(reward)
            episode_done += 1
        return root_node

    def choose(self, simulator, state, agent_id):
        legal_actions = simulator.getLegalActions(state, agent_id)
        return random.choice(legal_actions)

    def simulate(self, simulator, node, agent_id):
        state = copy.deepcopy(node.state)
        agent_index = agent_id
        sum_reward = 0
        depth = 0
        while depth < 1:
            action = self.choose(simulator, state, agent_index)
            state, reward = simulator.generateSuccessorWithReward(state, action, agent_index, True)
            simulator.current_agent_index = agent_index

            sum_reward += reward * pow(0.9, depth)

            if simulator.endState(state):
                return sum_reward

            depth += 1

        return sum_reward

class TrainingSequenceState(SequenceState):
    # 假定为其他三位玩家发牌，形成完整的游戏信息（当前board，4人手牌，draft，底池，垃圾池）
    def __init__(self, state):
        self.deck = state.deck
        self.board = state.board
        self.agents = state.agents
        self.board.draft = state.board.draft

        # 获取全部的牌
        cards = [(r + s) for r in ['2', '3', '4', '5', '6', '7', '8', '9', 't', 'j', 'q', 'k', 'a'] for s in
                 ['d', 'c', 'h', 's']]
        cards = cards * 2  # Sequence uses 2 decks.
        random.shuffle(cards)

        # 去掉我的手牌
        my_hand = None
        for agent_state in self.agents:
            if hasattr(agent_state, 'hand'):
                my_hand = agent_state.hand
        for card in my_hand:
            if card in cards:
                cards.remove(card)

        # 去掉所有的draft牌
        for card in self.board.draft:
            if card in cards:
                cards.remove(card)

        # 去掉所有的discard牌
        for card in self.deck.discards:
            if card in cards:
                cards.remove(card)

        self.deck.cards = cards

        # 为剩下三位玩家发牌
        for i, agent_state in enumerate(self.agents):
            if not hasattr(agent_state, 'hand'):
                self.agents[i].hand = self.deck.deal(6)

class TrainingSequenceGameRule(SequenceGameRule):
    # 用于创建基于当前游戏状态的游戏rules
    def __init__(self, state, agent_index, num_of_agent=4):
        super(TrainingSequenceGameRule, self).__init__(num_of_agent)
        self.perfect_information = True
        self.current_agent_index = agent_index
        self.num_of_agent = num_of_agent
        self.current_game_state = TrainingSequenceState(state)
        self.action_counter = 0

    # def trainUpdate(self, state, action, agent_index, need_reward):
    #     if need_reward:
    #         new_state, reward = self.generateSuccessorWithReward(state, action, agent_index)
    #     else:
    #         new_state = self.generateSuccessor(state, action, agent_index)
    #         reward = 0
    #
    #     current_agent_index = self.getNextAgentIndex() if action['type'] != 'trade' else self.current_agent_index
    #     return new_state, reward, current_agent_index

    def endState(self, state):
        scores = {RED:0, BLU:0}
        for plr_state in state.agents:
            scores[plr_state.colour] += plr_state.completed_seqs
        return scores[RED]>=2 or scores[BLU]>=2 or len(state.board.draft)==0

    def generateSuccessorWithReward(self, state, action, agent_id, simulate):
        cal_reward_obj = CalReward()
        state.board.new_seq = False
        plr_state = state.agents[agent_id]
        plr_state.last_action = action  # Record last action such that other agents can make use of this information.
        reward = 0
        draft_rwd = 0
        place_rwd = 0
        rm_rwd = 0
        card = action['play_card']
        draft = action['draft_card']
        if card:
            plr_state.hand.remove(card)  # Remove card from hand.
            plr_state.discard = card  # Add card to discard pile.
            state.deck.discards.append(
                card)  # Add card to global list of discards (some agents might find tracking this helpful).
            state.board.draft.remove(draft)  # Remove draft from draft selection.
            plr_state.hand.append(draft)  # Add draft to player hand.
            state.board.draft.extend(state.deck.deal())  # Replenish draft selection.

        if action['type'] == 'trade':
            plr_state.trade = True
            return state, 0

        # Update Sequence board. If action was to place/remove a marker, add/subtract it from the board.
        r, c = action['coords']
        if action['type'] == 'place':
            place_rwd = cal_reward_obj.calRewardFromBoard(action['coords'], state.board, plr_state)
            state.board.chips[r][c] = plr_state.colour
            state.board.empty_coords.remove(action['coords'])
            state.board.plr_coords[plr_state.colour].append(action['coords'])
        elif action['type'] == 'remove':
            state.board.chips[r][c] = EMPTY
            state.board.empty_coords.append(action['coords'])
            state.board.plr_coords[plr_state.opp_colour].remove(action['coords'])
            rm_rwd = cal_reward_obj.calRewardFromBoard((r, c), state.board, plr_state)
        else:
            print("Action unrecognised.")

        if draft in ['jd', 'jc'] and not simulate:
            draft_rwd += 100
        elif draft in ['jh', 'js'] and not simulate:
            draft_rwd += 40
        else:
            d_rwd = cal_reward_obj.calRewardFromDraft(draft, copy.deepcopy(state.board), plr_state)
            draft_rwd += d_rwd

        # if draft in ['2h', '3h', '4h', '5h']:
        #     draft_rwd += 5

        # Check if a sequence has just been completed. If so, upgrade chips to special sequence chips.
        if action['type'] == 'place':
            seq, seq_type = self.checkSeq(state.board.chips, plr_state, (r, c))
            if seq:
                seq_num = seq['num_seq']
                reward += seq_num * 20
                state.board.new_seq = seq_type
                for sequence in seq['coords']:
                    for r, c in sequence:
                        if state.board.chips[r][c] != JOKER:  # Joker spaces stay jokers.
                            state.board.chips[r][c] = plr_state.seq_colour
                            try:
                                state.board.plr_coords[plr_state.colour].remove(action['coords'])
                            except:  # Chip coords were already removed with the first sequence.
                                pass
                plr_state.completed_seqs += seq['num_seq']
                plr_state.seq_orientations.extend(seq['orientation'])

        reward += draft_rwd + place_rwd + rm_rwd
        plr_state.trade = False
        plr_state.agent_trace.action_reward.append((action, reward))  # Log this turn's action and any resultant score.
        plr_state.score += reward
        return state, reward

class CalReward:

    def calRewardFromBoard(self, position, board, plr_state):
        place_reward = 0
        clr = plr_state.colour
        sclr = plr_state.seq_colour
        opp_clr = plr_state.opp_colour
        opp_sclr = plr_state.opp_seq_colour
        lr, lc = position
        board.chips[lr][lc] = 'm'

        # c = 0
        # for (x, y) in [(4, 4), (4, 5), (5, 4), (5, 5)]:
        #     if board.chips[x][y] == clr or board.chips[x][y] == sclr or board.chips[x][y] == 'm':
        #         c += 1
        # if c == 4:
        #     place_reward += 80

        if position in [(4, 4), (4, 5), (5, 4), (5, 5)]:
            place_reward += 15
        # if lr >= 2 and lr <= 7 and lc >= 2 and lc <= 7 :
        #     place_reward += 1


        vr = []
        hz = []
        d1 = []
        d2 = []

        for i in range(10):
            vr.append((lr, i))
            hz.append((i, lc))

        j = -8
        while j < 9:
            d1.append((lr + j, lc + j))
            d2.append((lr + j, lc - j))
            j += 1
        d1 = [i for i in d1 if 0 <= min(i) and 9 >= max(i)]
        d2 = [i for i in d2 if 0 <= min(i) and 9 >= max(i)]

        for r, c in COORDS['jk']:
            board.chips[r][c] = clr


        for seq in [vr, hz, d1, d2]:
            if len(seq) < 5:
                continue

            chip_str = ''.join([board.chips[r][c] for (r, c) in seq])
            max_seq = 0
            last_max_seq = 0
            sequence_len = 0
            for i in range(len(chip_str)):
                if chip_str[i] == clr or chip_str[i] == sclr:
                    sequence_len += 1
                    if sequence_len > last_max_seq:
                        last_max_seq = sequence_len
                else:
                    sequence_len = 0
                if sequence_len >= 5:
                    break

            sequence_len = 0
            max_i = 0
            for i in range(len(chip_str)):
                if chip_str[i] == clr or chip_str[i] == sclr or chip_str[i] == 'm':
                    sequence_len += 1
                    if sequence_len > max_seq:
                        max_seq = sequence_len
                        max_i = i
                else:
                    sequence_len = 0
                if sequence_len >= 5:
                    break

            if max_seq > last_max_seq:
                rwd_me = 0
                if max_seq == 5:
                    rwd_me = 20
                else:
                    room = 0
                    potential = 0
                    if max_i - max_seq >= 0 and chip_str[max_i - max_seq] == '_':
                        room += 1
                        wr, wc = seq[max_i - max_seq]
                        wait_card = BOARD[wr][wc]
                        if wait_card in plr_state.hand:
                            potential += 1
                    if max_i + 1 < len(chip_str) and chip_str[max_i + 1] == '_':
                        room += 1
                        wr, wc = seq[max_i - max_seq]
                        wait_card = BOARD[wr][wc]
                        if wait_card in plr_state.hand:
                            potential += 1
                    if max_seq == 4:
                        if potential > 0:
                            rwd_me = 20
                        elif room == 2:
                            rwd_me = 16
                        elif room == 1:
                            rwd_me = 12
                    if max_seq == 3:
                        if potential > 0:
                            rwd_me = 8
                        elif room == 2:
                            rwd_me = 5
                        elif room == 1:
                            rwd_me = 3
                    if max_seq == 2:
                        rwd_me = 2
                place_reward += rwd_me


        # 对于敌方，以堵住为准
        for r, c in COORDS['jk']:
            board.chips[r][c] = opp_clr

        for seq in [vr, hz, d1, d2]:
            if len(seq) < 5:
                continue

            chip_str = ''.join([board.chips[r][c] for (r, c) in seq])
            max_seq = 0
            last_max_seq = 0
            sequence_len = 0
            for i in range(len(chip_str)):
                if chip_str[i] == opp_clr or chip_str[i] == opp_sclr:
                    sequence_len += 1
                    if sequence_len > last_max_seq:
                        last_max_seq = sequence_len
                else:
                    sequence_len = 0
                if sequence_len >= 5:
                    break

            sequence_len = 0
            max_i = 0
            for i in range(len(chip_str)):
                if chip_str[i] == opp_clr or chip_str[i] == 'm' or chip_str[i] == opp_sclr:
                    sequence_len += 1
                    if sequence_len > max_seq:
                        max_seq = sequence_len
                        max_i = i
                else:
                    sequence_len = 0
                if sequence_len >= 5:
                    break

            if max_seq > last_max_seq:
                rwd_opp = 0
                if max_seq == 5:
                    rwd_opp = 30
                else:
                    room = 0
                    if max_i-max_seq >= 0 and chip_str[max_i-max_seq] == '_':
                        room += 1
                    if max_i+1 < len(chip_str) and chip_str[max_i+1] == '_':
                        room += 1

                    if max_seq == 4:
                        if room == 2:
                            rwd_opp = 12
                        elif room == 1:
                            rwd_opp = 10
                    if max_seq == 3:
                        rwd_opp = 3
                    # if max_i == 2 and room == 2:
                    #     rwd = 2
                place_reward += rwd_opp

        for r, c in COORDS['jk']:
            board.chips[r][c] = '#'
        board.chips[lr][lc] = '_'
        return place_reward

    def calRewardFromDraft(self, draft, board, plr_state):
        for card in plr_state.hand:
            if card == draft:
                continue
            for r, c in COORDS[card]:
                if board.chips[r][c] == '_':
                    board.chips[r][c] = plr_state.colour

        draft_rwd = 0
        rwd = 0
        for x,y in COORDS[draft]:
            if board.chips[x][y] == '_':
                rwd = self.calRewardFromBoard((x,y), board, plr_state)
            if rwd > draft_rwd:
                draft_rwd = rwd
        return draft_rwd

