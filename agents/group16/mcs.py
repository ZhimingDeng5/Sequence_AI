import copy
import random
import math

from template import Agent
from agents.group16.training import TrainingSequenceGameRule

class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        print("创建一个mcts")
        self.mcts_obj = MCTS(self.id)


    def SelectAction(self, actions, game_state):
        print("创建一个simulator")
        simulator = TrainingSequenceGameRule(copy.deepcopy(game_state), self.id, 4)
        state = copy.deepcopy(simulator.current_game_state)
        root_node = self.mcts_obj.mcts(simulator, state, self.id)
        return root_node.bestAction()

class Node:

    nextNodeID = 0

    def __init__(self, parent, state):
        # self.simulator = simulator
        self.parent = parent
        self.state = copy.deepcopy(state)
        self.id = Node.nextNodeID
        Node.nextNodeID += 1

        self.visits = 0
        self.value = 0.0

    def getValue(self):
        return self.value

class StateNode(Node):
    def __init__(self, parent, state, agent_id, reward=0):
        super().__init__(parent, state)
        self.children = {}
        self.reward = reward
        self.agent_id = agent_id
        print("创建一个node：", self.id)

    def isFullyExpanded(self, simulator, state):
        valid_actions = simulator.getLegalActions(state, self.agent_id)
        valid_actions = set([self.dict2tuple(a) for a in valid_actions])

        if len(valid_actions) == len(self.children):
            return True
        else:
            return False

    def select(self, simulator, state):
        if not self.isFullyExpanded(simulator, state):
            print("选择node", self.id)
            return self
        else:
            actions = list(self.children.keys())
            q_values = []
            for action_tuple in actions:
                q_values.append(self.children[action_tuple].getValue)
            random_number = random.random()
            if random_number < 0.1:
                selected_action = random.choice(actions)
            else:
                max_q_value = -1000
                best_actions = []
                for action_tuple, q in zip(actions, q_values):
                    q_value=int(q)
                    if q_value > max_q_value:
                        max_q_value = q_value
                        best_actions = [action_tuple]
                    elif q_value == max_q_value:
                        best_actions.append(action_tuple)
                selected_action = random.choice(best_actions)
            child_node = self.children[selected_action]
            return child_node.select(simulator, child_node.state)

    def dict2tuple(self, dict):
        return tuple(dict.items())

    def tuple2dict(self, tuple):
        return dict(tuple)

    def expand(self, simulator, state):
        legal_actions = simulator.getLegalActions(state, self.agent_id)
        legal_actions_tuple = set([self.dict2tuple(a) for a in legal_actions])
        valid_actions = legal_actions_tuple - self.children.keys()
        action = random.choice(list(valid_actions))
        action = self.tuple2dict(action)
        new_state, _ = simulator.generateSuccessor(state, action, self.agent_id)
        new_child = StateNode(self, copy.deepcopy(new_state), self.agent_id)
        print("expand了一个子node：", new_child.id)

        self.children[self.dict2tuple(action)] = new_child

        return new_child

    def backPropagate(self, reward):
        self.visits +=1
        self.value = self.value + ((self.reward + reward - self.value) / self.visits)
        if self.value != 0:
            print("更新node的value值：", self.id, self.value)

        if self.parent != None:
            reward *= 0.9
            if reward != 0:
                print("有parentnode，id和更新值为", self.parent.id, reward)
            self.parent.backPropagate(reward)

    def bestAction(self):
        q_values = {}
        for action in self.children.keys():
            q_values[(self.state, action)] = round(self.children[action].getValue(), 3)

        max_q = -1000
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
        print("创建一个rootnode")
        root_node = StateNode(parent=None, state=copy.deepcopy(state), agent_id = agent_id)
        episode_done = 0
        while episode_done < 30:
            print("----开始第n次episode：", episode_done)
            selected_node = root_node.select(simulator, copy.deepcopy(state))

            if not simulator.endState(selected_node.state):
                child = selected_node.expand(simulator, copy.deepcopy(selected_node.state))
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

        sum_reward = 0.0
        depth = 0

        while not simulator.endState(state):
            print("simulate for node", node.id)
            action = self.choose(simulator, state, agent_index)
            state, reward, agent_index = simulator.trainUpdate(state, action, agent_index)
            simulator.current_agent_index = agent_index

            sum_reward += math.pow(0.9, depth) * reward
            depth += 1

            if simulator.endState(state):
                print("**自己赢了----")
                break

            if depth > 2:
                print("**不成超过既定depth---", depth)
                break

            game_end = False
            while agent_index != agent_id:
                actions = simulator.getLegalActions(state, agent_index)
                selected_action = random.choice(actions)
                state, _, agent_index = simulator.trainUpdate(state, selected_action, agent_index)
                simulator.current_agent_index = agent_index

                if simulator.endState(state):
                    print("**别人赢了----")
                    game_end = True
                    break

            if game_end:
                break
        if sum_reward != 0:
            print("为节点simulate并计算reward：", node.id, sum_reward)
        return sum_reward
