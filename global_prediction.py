import torch
import numpy as np 
import pickle 
import json 
import os 
from typing import List, Dict, Tuple
from evaluate_sceneseg import *
import time
from torch.nn import Parameter
from torch import optim
from torch import autograd
import torch.nn as nn
import argparse

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


CACHE = {}
TOTAL_SIMILARITY_CACHE = {}
SUPERSHOT_MAPPING_DICT = {}

TOTAL_NUM_SHOTS = 0


# Used to do a baseline test to make sure similarity is computed and the matrix is filled appropriately
first_tensor = torch.Tensor([1,2,3,4]).unsqueeze(0)
second_tensor = torch.Tensor([5,6,7,8]).unsqueeze(0)
third_tensor = torch.Tensor([9, 10, 11, 12]).unsqueeze(0)
fourth_tensor = torch.Tensor([13, 14, 15, 16]).unsqueeze(0)
fifth_tensor = torch.Tensor([17, 18, 19, 20]).unsqueeze(0)

simple_tensor = [[first_tensor, first_tensor, first_tensor, first_tensor, 3], 
[second_tensor, second_tensor, second_tensor, second_tensor, 3], 
[third_tensor, third_tensor, third_tensor, third_tensor, 3]] 
#[fourth_tensor, fourth_tensor, fourth_tensor, fourth_tensor, 3],
#[fifth_tensor, fifth_tensor, fifth_tensor, fifth_tensor, 3]]

USE_CACHE_FLAG = True

# Ratio taken from https://github.com/AnyiRao/SceneSeg/blob/master/lgss/config/all.py
RATIO = [0.5, 0.2, 0.2, 0.1]



def clear_caches():
	"""
	Helper function to reset caches when calculating all similarities again
	"""
	global CACHE
	global TOP_DOWN_CACHE
	global TOTAL_SIMILARITY_CACHE
	CACHE = {}
	TOTAL_SIMILARITY_CACHE = {}



"""
Implementing the global optimization algorithm from https://arxiv.org/pdf/2004.02678.pdf
with details from supplementary materials https://anyirao.com/files/papers/cvpr2020scene_supp.pdf
"""

def sigmoid(z: float) -> float:
	"""
	Sigmoid implementation taken from https://stackoverflow.com/questions/60746851/sigmoid-function-in-numpy
	"""
	return 1/(1 + np.exp(-z))

def get_supershots(boundary_prediction: torch.Tensor, threshold: float) -> List[List[int]]:
	"""
	Converts boundary predictions into supershots
	"""
	global TOTAL_NUM_SHOTS
	global SUPERSHOT_MAPPING_DICT
	all_supershots = []
	new_supershot = []
	num_supershots_made = 0
	TOTAL_NUM_SHOTS = len(boundary_prediction) + 1
	for index in range(len(boundary_prediction)):
		new_supershot.append(index)
		if boundary_prediction[index].item() > threshold:
			all_supershots.append(new_supershot)
			SUPERSHOT_MAPPING_DICT[num_supershots_made] = new_supershot
			num_supershots_made += 1
			new_supershot = []
	all_supershots.append(new_supershot)
	SUPERSHOT_MAPPING_DICT[num_supershots_made] = new_supershot
	return all_supershots

def get_supershot_features(data: Dict, supershot: List[int], parameters: List[Parameter]) -> List[torch.Tensor]:
	"""
	Gets the features for a supershot given the shots in the supershot as well as the parameters for each shot
	"""
	place_data = torch.sum(data['place'][supershot] * parameters[0][supershot], dim=0).unsqueeze(0)
	cast_data = torch.sum(data['cast'][supershot]  * parameters[1][supershot], dim=0).unsqueeze(0)
	action_data = torch.sum(data['action'][supershot]  * parameters[2][supershot], dim=0).unsqueeze(0)
	audio_data = torch.sum(data['audio'][supershot]  * parameters[3][supershot], dim=0).unsqueeze(0)
	return [place_data, cast_data, action_data, audio_data, len(supershot)]


def get_similarity_supershots(supershot_1: List[torch.Tensor], supershot_2: List[torch.Tensor]) -> float:
	"""
	Calculates the cosine similarity between two supershots
	"""
	place_similarity = torch.cosine_similarity(supershot_1[0].to(device), supershot_2[0].to(device)) * RATIO[0]
	cast_similarity = torch.cosine_similarity(supershot_1[1].to(device), supershot_2[1].to(device)) * RATIO[1]
	action_similarity = torch.cosine_similarity(supershot_1[2].to(device), supershot_2[2].to(device)) * RATIO[2]
	audio_similarity = torch.cosine_similarity(supershot_1[3].to(device), supershot_2[3].to(device)) * RATIO[3]
	total_similarity = place_similarity + cast_similarity + action_similarity + audio_similarity
	return total_similarity


def get_max_and_average_similarity(supershot_1: List[torch.Tensor], supershot_collection: List[List[torch.Tensor]], index_1: int, offset: int) -> Tuple[float]:
	"""
	Calculates the max and average similarity between a supershot and a list of other supershots. In terms of paper notation
	the first supershot corresponds to C_{k} and the collection corresponds to P_{k}
	"""
	global CACHE
	global TOTAL_NUM_SHOTS

	max_similarity = float('-inf')
	average_similarity = 0
	num_supershots = 0
	total_length_shot = supershot_1[-1]
	if len(supershot_collection) == 0:
		return torch.Tensor([0.]), torch.Tensor([0.])
	for index_2, supershot_2 in enumerate(supershot_collection):
		total_length_shot += supershot_2[-1]
		num_supershots += 1
		# This is done in case index_2 is one less then what it should be because index_1 is no longer in the list
		if index_2 >= index_1 - offset:
			index_2 += 1
		index_2 += offset
		if (index_1, index_2) in CACHE and USE_CACHE_FLAG:
			new_similarity = CACHE[(index_1, index_2)]
		else:
			new_similarity = get_similarity_supershots(supershot_1, supershot_2)
			CACHE[(index_1, index_2)] = new_similarity.item()
			CACHE[(index_2, index_1)] = new_similarity.item()

		max_similarity = max(max_similarity, new_similarity)
		average_similarity += new_similarity

	average_similarity /=  num_supershots

	#average_similarity *= ( 1 - 0.5 * (total_length_shot / TOTAL_NUM_SHOTS) ** 2)
	#max_similarity *= ( 1 - 0.5 * (total_length_shot / TOTAL_NUM_SHOTS) ** 2)

	if isinstance(max_similarity, torch.Tensor):
		max_similarity = nn.Sigmoid()(max_similarity)
	else:
		max_similarity = sigmoid(max_similarity)

	return max_similarity, average_similarity


def get_total_similarity(supershot_collection: List[List[torch.Tensor]], offset: int) -> float:
	"""
	Gets the total similarity between a group of supershots. In paper notation this would be φk
	The offset is to make sure the cache is used properly
	"""

	global TOTAL_SIMILARITY_CACHE

	indices = tuple(range(offset, offset + len(supershot_collection)))

	if indices in TOTAL_SIMILARITY_CACHE and USE_CACHE_FLAG:
		return TOTAL_SIMILARITY_CACHE[indices]

	total_length_shot = 0

	total_similarity = 0 
	if len(supershot_collection) == 1:
		return torch.Tensor([0.])
	for index in range(len(supershot_collection)):

		cur_supershot = supershot_collection[index]
		new_supershot_collection = supershot_collection[:index] + supershot_collection[index + 1:]
		#new_supershot_collection = supershot_collection[index + 1:]
		new_max_similarity, new_average_similarity = get_max_and_average_similarity(cur_supershot, new_supershot_collection, index + offset, offset)
		total_similarity += new_max_similarity
		total_similarity += new_average_similarity

	if isinstance(total_similarity, torch.Tensor):
		total_similarity_item = total_similarity.item()
	else:
		total_similarity_item = total_similarity

	TOTAL_SIMILARITY_CACHE[indices] = total_similarity_item
	return total_similarity


def solve_dp_problem_chunks(supershot_collection: List[List[torch.Tensor]], chunk_size: int=150) -> List[List[int]]:
	i = 0
	all_chunks = []
	while i < len(supershot_collection):
		new_chunk = supershot_collection[i: min(len(supershot_collection), i + chunk_size)]
		i = min(len(supershot_collection), i + chunk_size)
		all_chunks.append(new_chunk)

	offset = 0
	all_supershots = []
	for index, chunk in enumerate(all_chunks):
		solution_matrix, position_matrix =  solve_dp_problem_helper(chunk)
		new_supershots = dp_backtrack(chunk, solution_matrix, position_matrix)
		for first_index in range(len(new_supershots)):
			for second_index in range(len(new_supershots[first_index])):
				new_supershots[first_index][second_index] += offset
		all_supershots += new_supershots
		offset += len(chunk)
	return all_supershots



 
def solve_dp_problem_helper(supershot_collection: List[List[torch.Tensor]]) -> Tuple[np.array]:
	"""
	Takes in a group of supershots and computes the similarity matrix as well as the position matrix for backtracking
	"""
	clear_caches()
	j = len(supershot_collection) - 1
	solution_matrix = np.zeros((j, len(supershot_collection)))
	position_matrix = np.zeros((j, len(supershot_collection)))

	# First we get the similarity for the first supershot + second supershot, 1 + 2 + 3 supershot, ...
	for k in range(1, len(supershot_collection)):
		new_similarity = get_total_similarity(supershot_collection[: k + 1], 0)
		if isinstance(new_similarity, torch.Tensor):
			new_similarity = new_similarity.item()
		solution_matrix[0][k] = new_similarity

	for i in range(1, j):
		for k in range(0, len(supershot_collection)):

			max_l = float('-inf')
			new_value = float('-inf')
			if k < 1:
				max_l, new_value = 0,0
			for l in range(0, k):
				new_similarity = get_total_similarity(supershot_collection[l + 1: k + 1], l + 1)
				if isinstance(new_similarity, torch.Tensor):
					new_similarity = new_similarity.item()
				cur_value = solution_matrix[i - 1][l] + new_similarity
				new_value = max(new_value,cur_value )
				if new_value == cur_value:
					max_l = l

			solution_matrix[i][k] = new_value
			position_matrix[i][k] = max_l

	return solution_matrix, position_matrix


def dp_backtrack(supershot_collection:  List[List[torch.Tensor]], solution_matrix: np.array, position_matrix: np.array) -> List[List[int]]:
	"""
	Backtracks through the solution and position matrix to get the new list of supershots
	"""

	j = np.argmax(solution_matrix[:,solution_matrix.shape[1]-1])

	max_l = int(position_matrix[j][position_matrix.shape[1] - 1])

	all_supershots = []

	new_supershot = []
	for index in range(max_l, len(supershot_collection)):
		new_supershot.append(index)

	all_supershots.append(new_supershot)

	j -= 1

	while j > 1 and max_l > 0:

		new_max_l = int(position_matrix[j - 1][max_l])


		new_supershot = []
		for index in range(new_max_l, max_l):
			new_supershot.append(index)
		all_supershots.append(new_supershot)
		max_l = new_max_l

		j -= 1

	if len(all_supershots) < np.argmax(solution_matrix[:,solution_matrix.shape[1]-1]):
		new_supershot = []
		for i in range(max_l):
			new_supershot.append(i)
		all_supershots.append(new_supershot)

	return all_supershots


def get_tensor_supershots(data: Dict, supershot_collection: List[List[int]], parameters: List[Parameter]) -> List[List[int]]:
	"""
	Converts supershot indices into tensor features
	"""
	tensor_supershots = []
	for supershot in supershot_collection:
		new_tensor_supershot = get_supershot_features(data, supershot, parameters)
		tensor_supershots.append(new_tensor_supershot)
	return tensor_supershots
	#return solution_matrix_backtrack(supershot_collection, solution_matrix)

def convert_supershots_to_predictions(data: dict, supershot_collection: List[List[int]]) -> torch.Tensor:
	"""
	Converts supershots into predictions
	"""
	prediction_tensor = data['scene_transition_boundary_prediction'].float()
	last_shots = [elem[-1] for elem in supershot_collection]
	for index in range(len(prediction_tensor)):
		if index in last_shots:
			prediction_tensor[index] = 1.
		else:
			prediction_tensor[index] = prediction_tensor[index].item()
	return prediction_tensor


def convert_supershot_collation_to_supershots(supershot_collation: List[List[int]]) -> List[List[int]]:
	"""
	Converts a collection of supershots (the result of the DP) into  a collection of supershots based on the original shots
	"""
	global SUPERSHOT_MAPPING_DICT
	new_supershot_mapping_dict = {}
	all_supershots = []
	for index, collation in enumerate(supershot_collation):
		new_supershot = []
		for elem in collation:
			new_supershot += SUPERSHOT_MAPPING_DICT[elem]
		new_supershot_mapping_dict[index] = new_supershot
		all_supershots.append(new_supershot)
	SUPERSHOT_MAPPING_DICT = new_supershot_mapping_dict
	return all_supershots


def get_all_scores(gt_dict, pred_dict, shot_to_end_frame_dict):
	"""
	Helper to call the evaluation code
	"""
	scores = dict()
	scores["AP"], scores["mAP"], _ = calc_ap(gt_dict, pred_dict)
	scores["Miou"], _ = calc_miou(gt_dict, pred_dict, shot_to_end_frame_dict)
	scores["Precision"], scores["Recall"], scores["F1"], *_ = calc_precision_recall(gt_dict, pred_dict)
	print("Scores:", json.dumps(scores, indent=4))
	return scores["mAP"], scores["Miou"]


def baseline_test(data: dict, threshold: float):
	"""
	Code to optimize an instance
	"""
	print("CALCULATING INITIAL SCORE")
	pred =data['scene_transition_boundary_prediction'].float() 
	#pred = pred.float() > 0.5 

	pred_dict = {'file': pred}
	gt_dict = {'file': data['scene_transition_boundary_ground_truth']}
	shot_to_end_frame_dict = {}
	shot_to_end_frame_dict['file'] = data['shot_end_frame']

	init_mAP, init_Miou = get_all_scores(gt_dict, pred_dict, shot_to_end_frame_dict)
	all_supershots = get_supershots(data['scene_transition_boundary_prediction'], threshold)

	parameters = initialize_parameters(data)

	counter = 3

	while counter > 0 and len(all_supershots) > 1:

		tensor_supershots = get_tensor_supershots(data, all_supershots, parameters)
		solution_matrix, position_matrix  = solve_dp_problem_helper(tensor_supershots)
		supershot_collation = dp_backtrack(tensor_supershots, solution_matrix, position_matrix)
		#supershot_collation = solve_dp_problem_chunks(tensor_supershots)
		all_supershots = convert_supershot_collation_to_supershots(supershot_collation)
		new_supershot_predictions = convert_supershots_to_predictions(data, all_supershots)
		pred_dict = {'file': new_supershot_predictions}

		# Only update parameters if its not last loop
		if counter > 0 and len(all_supershots) > 1:
			parameters = run_optimization_step(data, all_supershots, supershot_collation, parameters, max_iters=1000)

	final_mAP, final_Miou = get_all_scores(gt_dict, pred_dict, shot_to_end_frame_dict)
	print('**' * 80)
	return final_mAP - init_mAP, final_Miou - final_Miou
	#parameters = run_optimization_step(data, all_supershots,  supershot_collation, parameters)


def initialize_parameters(data: dict):
	"""
	Initializes parameters to 1 uniformally before later optimizaiton
	"""
	num_shots = len(data['place'])
	place_parameters = Parameter(torch.ones(data['place'].shape))
	cast_parameters = Parameter(torch.ones(data['cast'].shape))
	action_parameters = Parameter(torch.ones(data['action'].shape))
	audio_parameters = Parameter(torch.ones(data['audio'].shape))
	return [place_parameters, cast_parameters, action_parameters, audio_parameters]


def run_optimization_step(data:dict, all_supershots: List[List[int]], supershot_collation: List[List[int]], parameters: List[Parameter], max_iters: int=300) -> List[Parameter]:
	"""
	Optimizes parameters for subsequent supershot representations
	"""
	optimizer = optim.SGD([
                {'params': parameters[0]},
                {'params': parameters[1]},
                {'params': parameters[2]},
                {'params': parameters[3]},
            ], lr=1e-1, momentum=0.9)

	for iteration in range(max_iters):
		old_parameters = torch.clone(parameters[0])
		total_score = 0
		tensor_supershots = get_tensor_supershots(data, all_supershots, parameters)
		#for elem in tensor_supershots:
		#	for sub_elem in elem:
		#		sub_elem = Variable(sub_elem.data, requires_grad=True)
		offset = 0
		clear_caches()
		for supershot_group in supershot_collation:
			total_score += get_total_similarity(tensor_supershots[offset: offset + len(supershot_group)], offset)
			offset += len(supershot_group)

		#print('total score: ', total_score)

		total_score *= -1
		total_score.backward()
		#autograd.backward([total_score], [parameters[0], parameters[1], parameters[2], parameters[3]], 
		#	grad_tensors=[torch.ones_like(parameters[0]), torch.ones_like(parameters[1]), torch.ones_like(parameters[2]), torch.ones_like(parameters[3])])
		optimizer.step()
		optimizer.zero_grad()

	return parameters


def basic_test():
	"""
	Runs a sanity test to make sure matrix is being calculated correctly
	"""
	global TOTAL_NUM_SHOTS
	TOTAL_NUM_SHOTS = sum(elem[-1] for elem in simple_tensor)
	solution_matrix, position_matrix  = solve_dp_problem_helper(simple_tensor)
	supershot_collation = dp_backtrack(simple_tensor, solution_matrix, position_matrix)
	print('supershot collation: ', supershot_collation)
	all_supershots = convert_supershot_collation_to_supershots(supershot_collation)
	print('all supershots: ', all_supershots)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Arguments for running optimization step')
	parser.add_argument("--data_directory", type=str, help = "Directory with data")
	args = parser.parse_args()

	total_mAP, total_Miou = 0, 0
	counter = 0

	for file in os.listdir(args.data_directory):
		if file.endswith('.pkl'):
			new_file = os.path.join(args.data_directory, file)
		with open(new_file, 'rb') as f:
			data = pickle.load(f)

		new_mAP, new_Miou = baseline_test(data, 0.4)
		total_mAP += new_mAP
		total_Miou += new_Miou
		counter += 1

	total_mAP /= counter
	total_Miou /= counter
	print('Average mAP difference: ', total_mAP)
	print('Average Miou difference: ', total_Miou)



	#with open('/Users/humzaiqbal/Downloads/data/tt0078788.pkl', 'rb') as f:
	#	data = pickle.load(f)

	#basic_test()

	#baseline_test(data, 0.4)

"""
[1, 2, 3, 4], [5,6,7, 8], [9, 10, 11, 12]


pairwise similarities: (1, 2) -> tensor([0.9689])
					   (2, 3) -> tensor([0.9979])
					   (3, 1) -> tensor([0.9510])

subproblems = [0, 0, 0
               0, 0, 0]



"""



