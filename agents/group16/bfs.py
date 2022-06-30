import copy
import random
import time

from Sequence.sequence_model import SequenceState, SequenceGameRule
from Sequence.sequence_utils import EMPTY,JOKER,  RED, BLU

from template import Agent


class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)

    def bfs(self, beginsta, agent_id):
        searchdrua = time.time()

        # define goal state
        def checkgoal(state):

            positions = state.board.plr_coords[state.agents[agent_id].colour]
            spositions = beginsta.board.plr_coords[state.agents[agent_id].colour]

            if len(positions) == 0:
                return True

            for next1pos in spositions:
                for next2pos in positions:
                    if next2pos not in spositions:
                        if (next1pos[1] == next2pos[1] or next1pos[0] == next2pos[0]
                                or abs(next1pos[1] - next2pos[1])==abs(next1pos[0] - next2pos[0])) \
                                and abs(next1pos[1] - next2pos[1])+abs(next1pos[0] - next2pos[0])<=4:
                            return True

            return False

        sim = SimulateGame(copy.deepcopy(beginsta), self.id, 4)
        state = sim.current_game_state

        lis = [(state, [])]
        while len(lis) > 0:
            statenow, action_lis = lis.pop(0)
            valid_actions = sim.getLegalActions(statenow, agent_id)

            for action in valid_actions:
                nstate, reward, cur_agent_index = sim.nextState(copy.deepcopy(statenow), action, self.id)

                if checkgoal(nstate) or searchdrua > 0.8:
                    return action_lis + [action]
                lis.append((nstate, action_lis + [action]))

        return None

    def SelectAction(self, actions, game_state):
        nextactions = self.bfs(game_state, self.id)  # Action
        if len(nextactions) == 0:
            return random.choice(actions)
        return nextactions[0]
    # define a simulate game and get state


class SimulateGameState(SequenceState):
    def __init__(self, state):
        self.deck = state.deck
        self.board = state.board
        self.board.draft = state.board.draft
        self.agents = state.agents
        usedcard = [(r + s) for r in ['2', '3', '4', '5', '6', '7', '8', '9', 't', 'j', 'q', 'k', 'a'] for s in
                    ['d', 'c', 'h', 's']] * 2
        current_agent_hand = None

        for a in self.agents:
            if hasattr(a, 'hand'):
                current_agent_hand = a.hand
        # remove current agent's hand from deck.cards
        for c in current_agent_hand:
            if c in usedcard:
                usedcard.remove(c)
        # remove  discard cards from deck.cards
        for c in self.deck.discards:
            if c in usedcard:
                usedcard.remove(c)
        # remove cards in draft from deck.cards
        for c in self.board.draft:
            if c in usedcard:
                usedcard.remove(c)

        self.deck.cards = usedcard
        # give the other agents cards for simulation
        for i, agent_state in enumerate(self.agents):
            if not hasattr(agent_state, 'hand'):
                self.agents[i].hand = self.deck.deal(6)


class SimulateGame(SequenceGameRule):
    # use the created simulating game
    def __init__(self, state, agent_index, num_of_agent=4):
        super().__init__(num_of_agent)
        self.perfect_information = True
        self.current_game_state = SimulateGameState(state)
        self.current_agent_index = agent_index
        self.num_of_agent = num_of_agent
    # this part might be similar to SequenceGameRule.generateSuccessor function in sequence_model.py
    def generateSuccessor(self, state, action, id):
        state.board.new_seq = False

        plr_state = state.agents[id]
        plr_state.last_action = action
        reward = 0

        c = action['play_card']
        d = action['draft_card']
        if c:
            plr_state.hand.remove(c)
            plr_state.discard = c
            state.deck.discards.append(
                c)
            state.board.draft.remove(d)
            plr_state.hand.append(d)
            state.board.draft.extend(state.deck.deal())

        if action['type'] == 'trade':
            plr_state.trade = True
            return state, 0

        r, c = action['coords']
        if action['type'] == 'place':
            state.board.chips[r][c] = plr_state.colour
            state.board.empty_coords.remove(action['coords'])
            state.board.plr_coords[plr_state.colour].append(action['coords'])
        elif action['type'] == 'remove':
            state.board.chips[r][c] = EMPTY
            state.board.empty_coords.append(action['coords'])
        else:
            print("Action unrecognised.")

        if action['type'] == 'place':
            seq, seq_type = self.checkSeq(state.board.chips, plr_state, (r, c))
            if seq:
                reward += seq['num_seq']
                state.board.new_seq = seq_type
                for lis in seq['coords']:
                    for r, c in lis:
                        if state.board.chips[r][c] != JOKER:
                            state.board.chips[r][c] = plr_state.seq_colour
                            try:
                                state.board.plr_coords[plr_state.colour].remove(action['coords'])
                            except:
                                pass
                plr_state.completed_seqs += seq['num_seq']
                plr_state.seq_orientations.extend(seq['orientation'])

        plr_state.trade = False
        plr_state.agent_trace.action_reward.append((action, reward))
        plr_state.score += reward
        return state, reward

    # execute game for a turn of play and return with next state
    def nextState(self, state, action, agent_index):
        newState, reward = self.generateSuccessor(state, action, agent_index)
        current_agent_index = self.getNextAgentIndex() if action['type'] != 'trade' else self.current_agent_index

        return newState, reward, current_agent_index

    # check if it is the end state
    def isendState(self, state):
        scores = {RED: 0, BLU: 0}
        for plr_state in state.agents:
            scores[plr_state.colour] += plr_state.completed_seqs
        return scores[RED] >= 2 or scores[BLU] >= 2 or len(state.board.draft) == 0