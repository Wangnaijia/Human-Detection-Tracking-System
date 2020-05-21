/home/wnj/venvs/ven27/bin/python2.7 /home/wnj/projects/human-detector/object_detector/extract_features.py
Calculating the descriptors for the positive samples and saving
Positive features 2416 saved in ../data/features/pos.
Calculating the descriptors for the negative samples and saving
Negative features 12180 saved in ../data/features/neg.
Calculating the descriptors for test samples and saving
Test features 1126 saved in ../data/features/test.
Completed calculating in 62.918408 seconds

Process finished with exit code 0


(14596, 3780) 14596
Training a Linear SVM Classifier
Classifier saved to ../data/models/svm_new_12180neg.model
The cast of time is 2.97918581963 seconds

Process finished with exit code 0


1126pos:

 The accuracy is:
0.793072824156
 Time cost is:
1.70381999016




 The pos detected:
893
 The pos total:
1126
 The neg detected:
1187
 The neg total:
1200
 Time cost is:
0.403949975967


---------------------2436neg-----------
neg1
/train_svm.py
(4852, 3780) 4852
Training a Linear SVM Classifier
Classifier saved to ../data/models/svm_new_12180neg.model
The cast of time is 1.2367059803 seconds

Process finished with exit code 0

neg2
/train_svm.py
(4916, 3780) 4916
Training a Linear SVM Classifier
Classifier saved to ../data/models/svm_cross.model
The cast of time is 1.15983700752 seconds

Process finished with exit code 0

neg3
(4976, 3780) 4976
Training a Linear SVM Classifier
Classifier saved to ../data/models/svm_cross.model
The cast of time is 1.27022600174 seconds

Process finished with exit code 0

neg4
/train_svm.py
(5046, 3780) 5046
Training a Linear SVM Classifier
Classifier saved to ../data/models/svm_cross.model
The cast of time is 1.4921810627 seconds

Process finished with exit code 0

neg5
/train_svm.py
(5085, 3780) 5085
Training a Linear SVM Classifier
Classifier saved to ../data/models/svm_cross.model
The cast of time is 1.56142997742 seconds

Process finished with exit code 0


 The pos detected:
912
 The pos total:
1126
 The neg detected:
1185
 The neg total:
1200
 Time cost is:
0.396106958389

-------------------------PCA----------------------------
start to do PCA
(4852, 3780)
PCA takes 12.485366 seconds
(4852, 500) 4852
Training a Linear SVM Classifier

Classifier saved to ../data/models/

Process finished with exit code 0

感觉博主写的PCA+SVM训练和检测过程是没问题的，我最近实验了一下，不知道是不是提取的HOG特征正负样本距离太近，PCA+SVM检测的打标与训练的正负样本数比例完全挂钩，就是哪种样本数多一点，检测时就会全标记为一种;如果正负样本数一样多，那检测就是随心所欲打1打0了，真是百思不得其解，难道是说降维后数据重叠了吗？网上面很多PCA+SVM的案例都是直接对灰度值降维，对HOG特征降维实践起来还是没能成功，不知道论文里PCA降维到100维再训练检测这种都是怎么做到的，太难了。

