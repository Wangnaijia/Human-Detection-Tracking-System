pos_im_path = '../data/images/pos_person'
neg_im_path = '../data/images/neg_person'
test_im_path = '../data/images/negTest'
min_wdw_sz = [64, 128]
step_size = [8, 8]
orientations = 9
pixels_per_cell = [8, 8]
cells_per_block = [2, 2]
visualize = False
normalize = True
pos_feat_path = '../data/features/pos'
neg_feat_path = '../data/features/neg'
# for cross train
final_neg_feat_path = '../data/features/neg1'
model_path = '../data/models/'
pca_path = '../data/PCA'
threshold = .3
# for INRIAN testing
test_feat_path_pos = '../data/features/testPos'
test_feat_path_neg = '../data/features/testNeg'
# for saving results
result_path = '../data/results'

flagCrop = False
flagExtract = False
flagTrain = False
flagHardValidate = False
flagClassifierTest = False
flagPSO = False
