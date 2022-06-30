from Sequence.sequence_model import SequenceGameRule, SequenceState
from Sequence.sequence_utils import *
import random

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

    def generateSuccessor(self, state, action, agent_id):
        state.board.new_seq = False
        plr_state = state.agents[agent_id]
        plr_state.last_action = action  # Record last action such that other agents can make use of this information.
        reward = 0

        # Update agent state. Take the card in play from the agent, discard, draw the selected draft, deal a new draft.
        # If agent was allowed to trade but chose not to, there is no card played, and hand remains the same.
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

        # If action was to trade in a dead card, action is complete, and agent gets to play another card.
        if action['type'] == 'trade':
            plr_state.trade = True  # Switch trade flag to prohibit agent performing a second trade this turn.
            return state, 0

        # Update Sequence board. If action was to place/remove a marker, add/subtract it from the board.
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

        # Check if a sequence has just been completed. If so, upgrade chips to special sequence chips.
        if action['type'] == 'place':
            seq, seq_type = self.checkSeq(state.board.chips, plr_state, (r, c))
            if seq:
                reward += seq['num_seq'] *10
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

        plr_state.trade = False  # Reset trade flag if agent has completed a full turn.
        plr_state.agent_trace.action_reward.append((action, reward))  # Log this turn's action and any resultant score.
        plr_state.score += reward
        return state, reward

    def trainUpdate(self, state, action, agent_index):
        temp_state = state;
        new_state, reward = self.generateSuccessor(state, action, agent_index)
        current_agent_index = self.getNextAgentIndex() if action['type'] != 'trade' else self.current_agent_index
        self.action_counter += 1

        return new_state, reward, current_agent_index

    def endState(self, state):
        scores = {RED:0, BLU:0}
        for plr_state in state.agents:
            scores[plr_state.colour] += plr_state.completed_seqs
        return scores[RED]>=2 or scores[BLU]>=2 or len(state.board.draft)==0