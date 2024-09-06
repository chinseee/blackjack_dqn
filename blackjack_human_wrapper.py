import numpy as np
import re
from BlackjackEnv import BlackjackEnv
from QNet import QNet
from Agent import obs_to_tensor
import torch

card_helper_arr = np.array(['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'])
move_helper_arr = np.array(['hit', 'stand', 'double', 'split', 'surrender'])

env = BlackjackEnv()

obs, info = env.reset()

def parse_move(move_str: str):
    matches = []
    for i, move in enumerate(move_helper_arr):
        if re.match('^' + move_str.lower(), move):
            matches.append(i)
    return matches

net = QNet(79, 5)
net.load_state_dict(torch.load('q_net.pt'))
net.eval()

while True:
    print(f"n cards: {obs['n_cards']:2d}")
    print('cards remaining:')
    for n in card_helper_arr:
        print(f"{n:>4s}", end='')
    print()
    for n in obs['cards_remaining']:
        print(f"{n:4d}", end='')
    print()
    print(f"dealer card: {card_helper_arr[obs['dealer_card']]}")
    for i, hand in enumerate(obs['hands']):
        if np.sum(hand) != 0:
            print(f'hand {i + 1:1d}:')
            for n in card_helper_arr:
                print(f"{n:>4s}", end='')
            print()
            for n in hand:
                print(f"{n:4d}", end='')
            print()

    legal = env.get_legal()
    print('legal moves:', end='')
    for i, move in enumerate(move_helper_arr):
        if legal[i]:
            print('  ' + move, end='')
    print()

    action = 0
    
    net_obs = obs_to_tensor(obs)
    action_values: torch.Tensor = net(net_obs)
    action_values = action_values.cpu().data.numpy()
    action_values[~legal] = -np.inf
    print('ai recommends: ' + move_helper_arr[np.argmax(action_values)])

    print('your move:')
    while True:
        moves = parse_move(input())
        if len(moves) == 1:
            action = moves[0]
            break
        elif len(moves) > 1:
            print('ambiguous between:', end='')
            for i in moves:
                print('  ' + move_helper_arr[i], end='')
            print()
        else:
            print('invalid move')
    
    obs, rew, term, trunc, info = env.step(action)
    if 'round_over' in info.keys():
        print(f'round over, reward: {rew:3.1f}')