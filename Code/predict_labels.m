function [Y_hat] = predict_labels(word_counts, cnn_feat, prob_feat, color_feat, raw_imgs, raw_tweets)
% Inputs:   word_counts     nx10000 word counts features
%           cnn_feat        nx4096 Penultimate layer of Convolutional
%                               Neural Network features
%           prob_feat       nx1365 Probabilities on 1000 objects and 365
%                               scene categories
%           color_feat      nx33 Color spectra of the images (33 dim)
%           raw_imgs        nx30000 raw images pixels
%           raw_tweets      nx1 cells containing all the raw tweets in text
% Outputs:  Y_hat           nx1 predicted labels (1 for joy, 0 for sad)

%% SVM model
% liblinear
load modelSVM
addpath('./liblinear');
[~, ~, scoresSVM] = predict(ones(size(word_counts, 1), 1), sparse(word_counts), modelSVM);

% % matlab svm
% load modelSVM_mat
% [~, scoresSVM] = predict(modelSVM, full(word_counts));

%% Logistic regression
% Load the model
load modelLR

[~, ~, probLR] = predict(ones(size(word_counts, 1), 1), sparse(word_counts), modelLR, '-q  -b 1');

%% Naive Bayes
load modelNB

prob_feat = round(prob_feat*100);
[~, postProbNB ,~] = predict(modelNB, prob_feat);

%% Adaboost
load modelAda
addpath('./piotr_toolbox/toolbox/classify')

logRatio = double(adaBoostApply(single(cnn_feat), modelAda));

%% Ensemble all models
load modelDecision
Y_hat = predict(ones(size(word_counts, 1), 1), sparse([postProbNB(:,1) scoresSVM(:,1) probLR(:,1) logRatio]), modelDecision, '-q');
Y_hat = full(Y_hat);
end
