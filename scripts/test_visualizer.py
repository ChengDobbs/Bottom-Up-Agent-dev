import json
import requests
import cv2

from utils.utils import cv_to_base64
BASE_URL = 'http://127.0.0.1:5000'


def test_update_data_endpoint():
    im1 = cv2.imread("scripts/images/1.jpg")
    im2 = cv2.imread("scripts/images/2.jpg")
    im3 = cv2.imread("scripts/images/4.jpg")

    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2RGB)

    im1_base64 = cv_to_base64(im1)
    im2_base64 = cv_to_base64(im2)
    im3_base64 = cv_to_base64(im3)

    im1_data = f'data:image/png;base64,{im1_base64}'
    im2_data = f'data:image/png;base64,{im2_base64}'
    im3_data = f'data:image/png;base64,{im3_base64}'

    # Prepare and send payload to live server
    payload = {
        'run_states': {'potential_actions': 5, 'temperature': 0.5, 'decision': 'explore'},
        'action_goal': {'id': 1, 'name': 'Action', 'description': 'Desc'},
        'suspend_actions': [2],
        'candidate_actions': [{'id': 3, 'name': 'Cand', 'fitness': 0.0, 'num': 1, 'prob': 0.1}, 
                              {'id': 1, 'name': 'Cand', 'fitness': 1.9, 'num': 2, 'prob': 0.7}, 
                              {'id': 2, 'name': 'Cand', 'fitness': 1.0, 'num': 2, 'prob': 0.2}, 
                              {'id': 1, 'name': 'Cand', 'fitness': 1.9, 'num': 2, 'prob': 0.7}, 
                              {'id': 2, 'name': 'Cand', 'fitness': 1.0, 'num': 2, 'prob': 0.2}, 
                              {'id': 1, 'name': 'Cand', 'fitness': 1.9, 'num': 2, 'prob': 0.7}, 
                              {'id': 2, 'name': 'Cand', 'fitness': 1.0, 'num': 2, 'prob': 0.2}, 
                              {'id': 1, 'name': 'Cand', 'fitness': 1.9, 'num': 2, 'prob': 0.7}, 
                              {'id': 4, 'name': 'Cand', 'fitness': 1.0, 'num': 2, 'prob': 0.2}],
        'explore_tree': {'name': 'node', 'state': 'Existed', 'children': [
            {'name': 'child1', 'state': 'Potential', 'children': []},
            {'name': 'child2', 'state': 'Selected', 'children': [{'name': 'node', 'state': 'Existed'}, {'name': 'node', 'state': 'Potential'}, {'name': 'node', 'state': 'New'}]}
        ]},
        'delete_ids': [3],
        'exec_chain': [
            {'screen': im1_data, 'operation': {'operate': 'LeftSingleClick', 'params': {'x': 100, 'y': 200}}},
            {'screen': im2_data, 'operation': {'operate': 'RightSingleClick', 'params': {'x': 150, 'y': 250}}},
            {'screen': im3_data}
        ]
    }
    try:
        r = requests.post(f"{BASE_URL}/api/update", json=payload)
    except Exception as e:
        pass


if __name__ == '__main__':
    test_update_data_endpoint()


