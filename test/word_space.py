import torch

edge_score = torch.tensor([[1, 2, 3, 4, 5], [2, 2, 3, 0, 0], [3, 1, 2, 0, 0]])
adj_matrix = torch.tensor([[[1, 0, 0],
                            [0, 1, 1],
                            [0, 1, 1]],
                           [[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                           [[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]]])
adj_flatten = torch.flatten(adj_matrix, 1)
print(adj_flatten)
print(adj_matrix)
index = torch.tensor([[0, 4, 5, 7, 8],
                      [0, 4, 8, 1, 0],
                      [0, 4, 8, 1, 0],
                      ])
adj_flatten.scatter_(1, index, edge_score)
print(torch.reshape(adj_flatten,adj_matrix.shape))
"""
target[i][index[i][j]] = source[i][j]
[[0,4,5,6,7,8],
[0,4,8]
[0,4,8],
]
"""
