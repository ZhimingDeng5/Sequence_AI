import copy

from template import Agent
from agents.group16.training import TrainingSequenceGameRule, TrainingSequenceState
import random


class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.alpha = 0.5
        self.discount = 0.8
        self.epsilon = 0.2
        self.simulator = None
        # self.train_mode = True
        self.weights = {}

    def extract_features(self, game_state, action):
        current_seq = game_state.agents[self.id].completed_seqs + 1
        return {'current_seq': current_seq}

    def get_legal_actions(self, state):
        return self.simulator.getLegalActions(state, self.id)

    def get_q_value(self, state, action):
        features = self.extract_features(state, action)
        sum = 0
        for feature, f_value in features.items():
            sum += self.weights.get(feature, 0) * f_value
        return sum


    def update_weight_vector(self, state, action, next_state, reward):
        nextstep_actions = self.get_legal_actions(next_state)
        nextstep_q_values = [self.get_q_value(next_state, a) for a in nextstep_actions]
        if len(nextstep_q_values) == 0:
            max_q_next_step = 0
        else:
            max_q_next_step = max(nextstep_q_values)

        diff = reward + self.discount * max_q_next_step - self.get_q_value(state, action)
        features = self.extract_features(state, action)
        for feature_name, f_value in features.items():
            weight = self.weights.get(feature_name, 0)
            self.weights[feature_name] = weight + self.alpha * diff * f_value
        print(self.weights)

    def best_action(self, state):
        legalActions = self.get_legal_actions(state)

        actions = []
        q_values = []
        for a in legalActions:
            q_value = self.get_q_value(state, a)
            actions.append(a)
            q_values.append(q_value)
        best_actions = []
        for i, q in enumerate(q_values):
            if q == max(q_values):
                best_actions.append(actions[i])
        if len(best_actions) == 0:
            return None
        return random.choice(best_actions)

    def SelectAction(self, actions, game_state):
        self.simulator = TrainingSequenceGameRule(copy.deepcopy(game_state), self.id, 4)

        # if self.train_mode:
        training_num = 0
        while training_num < 2:
            if training_num < 100:
                self.epsilon = 0.2
            else:
                self.epsilon = 0.1
            current_agent_index = self.id
            current_state = self.simulator.current_game_state
            print("在training啦:", training_num, current_agent_index)
            i = 0
            while i < 8:
                print("在training里面出牌啦:", current_agent_index)
                if current_agent_index == self.id:
                    random_number = random.random()
                    if random_number < self.epsilon:
                        action = random.choice(self.get_legal_actions(current_state))
                    else:
                        action = self.best_action(current_state)
                else:
                    action = random.choice(self.simulator.getLegalActions(current_state, current_agent_index))

                new_state, reward, next_agent_index = \
                    self.simulator.trainUpdate(self.simulator.current_game_state, action, current_agent_index)
                self.simulator.current_agent_index = next_agent_index

                if current_agent_index == self.id:
                    self.update_weight_vector(current_state, action, new_state, reward)
                    print("是自己，更新weight啦")

                current_agent_index = next_agent_index
                print("下一个机器人准备", current_agent_index)
                current_state = new_state

                # if self.simulator.endState(current_state):
                #     break

                i += 1
            # self.train_mode = False
            training_num += 1

        # else:
        action = self.best_action(game_state)

        return action
